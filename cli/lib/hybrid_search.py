import os

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from constants import INDEX_FILE


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_FILE):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25Results = self._bm25_search(query, limit=500*limit)
        semanticResults = self.semantic_search.search_chunks(query, limit=500*limit)
        
        mixedResults: dict[int, dict] = {}
        maximum_score = float("-inf")
        minimum_score = float("inf")
        for (docId, score) in bm25Results:
            if score > maximum_score:
                 maximum_score = score
            if score < minimum_score:
                 minimum_score = score
            
            if docId not in mixedResults:
                mixedResults[docId] = {
                    "semantic_score": 0.0
                }
        
        for result in semanticResults:
            if result["id"] not in mixedResults:
                mixedResults[result["id"]] = {
                    "bm25_score": 0,
                }
            mixedResults[result["id"]]["semantic_score"] = result["score"]
        
        for (docId, score) in bm25Results:
            mixedResults[docId]["bm25_score"] = normalize(score, minimum_score, maximum_score)
            
        for id in mixedResults:
            semanticScore = mixedResults[id]["semantic_score"]
            keywordScore = mixedResults[id]["bm25_score"]
            mixedResults[id]["hybrid_score"] = hybrid_score(keywordScore, semanticScore, alpha)
            mixedResults[id]["data"] = self.idx.docmap[id]

        return sorted(mixedResults.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:limit]



    def rrf_search(self, query, k, limit=10):
        bm25Results = self._bm25_search(query, limit * 500)
        semanticResults = self.semantic_search.search_chunks(query, limit * 500)

        resultMap: dict[int, dict] = {}
        for i, (docId, score) in enumerate(bm25Results):
            if docId not in resultMap:
                resultMap[docId] = {
                    "rrf": 0,
                    "semantic_score": 0
                }

            doc = self.idx.docmap[docId]
            resultMap[docId]["title"] = doc["title"]
            resultMap[docId]["desc"] = doc["description"]
            resultMap[docId]["rrf"] += calculate_rrf(i+1, k)
            resultMap[docId]["bm25_score"] = score
        
        for i, data in enumerate(semanticResults):
            docId = data["id"]
            if docId not in resultMap:
                resultMap[docId] = {
                    "rrf": 0,
                    "bm25_score": 0
                }
            
            resultMap[docId]["title"] = data["title"]
            resultMap[docId]["desc"] = data["document"]
            resultMap[docId]["rrf"] += calculate_rrf(i+1, k)
            resultMap[docId]["semantic_score"] = data["score"]
        
        return sorted(resultMap.values(), key= lambda x: x["rrf"], reverse=True)[:limit]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(score, minimum_score, maximum_score):
    if minimum_score == maximum_score:
        return 1.0
    return (score - minimum_score) / (maximum_score - minimum_score)

def calculate_rrf(rank: int, k: int):
    return 1/(k + rank)