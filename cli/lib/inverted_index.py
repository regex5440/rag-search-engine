from pickle import dump, load
import os
from helpers import tokenizeSearchTerm, loadMovies
from collections import Counter
from math import log
import constants

class InvertedIndex():
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {} # tokens: str -> set[int] of docIds
        self.docmap: dict[int, dict] = {} # docId: int -> docObject
        self.term_frequency: dict[int, Counter] = {} # docId -> Counter
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenizeSearchTerm(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id].update([token])

    def get_tf(self, doc_id, term) -> int:
        tokens = tokenizeSearchTerm(term)
        if len(tokens) > 1:
            raise Exception("only singly word is expected")
        if doc_id not in self.term_frequency:
            return 0
        return self.term_frequency[doc_id].get(tokens[0]) or 0

    def get_documents(self, term: str):
        [token] = tokenizeSearchTerm(term)
        if token is None:
            return []
        if token not in self.index:
            return []
        
        return sorted(self.index[token])
    
    def build(self):
        moviesList = loadMovies()
        for m in moviesList:
            id = m["id"]
            self.docmap[id] = m
            self.__add_document(id, f"{m['title']} {m['description']}")

    def save(self):
        print("Saving Indexed data...")
        os.makedirs(constants.CACHE_FOLDER, exist_ok=True)

        with open(constants.INDEX_FILE, "wb") as f:
            dump(self.index, f)
            f.close()
            print("Saved index to", constants.INDEX_FILE)
        with open(constants.DOCUMENT_OBJ_FILE, "wb") as f:
            dump(self.docmap, f)
            f.close()
            print("Saved movie mapping to", constants.DOCUMENT_OBJ_FILE)
        with open(constants.TERM_FREQUENCY_FILE, "wb") as f:
            dump(self.term_frequency, f)
            f.close()
            print("Saved term frequencies to", constants.TERM_FREQUENCY_FILE)
        with open(constants.DOC_LENGTH_FILE, "wb") as f:
            dump(self.doc_lengths, f)
            f.close()
            print("Saved doc length to", constants.DOC_LENGTH_FILE)

    def load(self):
        if len(self.index) > 0 and len(self.docmap) > 0 and len(self.term_frequency) > 0:
            return
        if not os.path.exists(constants.CACHE_FOLDER):
            raise FileNotFoundError()
        with open(constants.INDEX_FILE, "rb") as f:
            self.index = load(f)
            f.close()
        with open(constants.DOCUMENT_OBJ_FILE, "rb") as f:
            self.docmap = load(f)
            f.close()
        with open(constants.TERM_FREQUENCY_FILE, "rb") as f:
            self.term_frequency = load(f)
            f.close()
        with open(constants.DOC_LENGTH_FILE, "rb") as f:
            self.doc_lengths = load(f)
            f.close()
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenizeSearchTerm(term)
        if len(tokens) > 1:
            raise Exception("only single term can be used")
        N = len(self.docmap)
        df = len(self.index[tokens[0]])
        return log((N - df + 0.5)/(df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=constants.BM25_K1, length_normalization_factor=constants.BM25_B):
        docLen = self.doc_lengths[doc_id]
        if docLen == None:
            docLen = 0.0
        leng_norm = 1 - length_normalization_factor + length_normalization_factor * (docLen/self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1))/(tf + k1 * leng_norm)
    
    def __get_avg_doc_length(self) -> float:
        sum = 0
        for doc_len in self.doc_lengths.values():
            sum += doc_len
        
        if sum == 0:
            return 0.0
        return sum/len(self.doc_lengths)
    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def bm25_search(self, query, limit):
        tokens = tokenizeSearchTerm(query)
        if len(tokens) == 0:
            return []
        
        matchingDocs = {}

        for token in tokens:
            for doc_id in self.index.get(token, []):
                if doc_id not in matchingDocs:
                    matchingDocs[doc_id] = 0
                matchingDocs[doc_id] += self.bm25(doc_id, token)
        
        return sorted(matchingDocs.items(), key=lambda item: item[1], reverse=True)[:limit]
