from cmd_parser import hybridSearchParser
from helpers import loadMovies
from lib.hybrid_search import HybridSearch
from lib.llm import LLM
from sentence_transformers import CrossEncoder

def main() -> None:
    parser = hybridSearchParser()

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.list
            if len(scores) == 0:
                return
            maximum_score = max(scores)
            minimum_score = min(scores)
            for i in scores:
                if minimum_score == maximum_score:
                    print("* 1.0")
                    continue
                score = (i - minimum_score) / (maximum_score - minimum_score)
                print(f"* {score:.4f}") 
        case "weighted-search":
            hrb = HybridSearch(loadMovies())
            results = hrb.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results):
                movieName = r[1]["data"]["title"]
                desc = r[1]["data"]["description"]
                semantic_score = r[1]["semantic_score"]
                keyword_score = r[1]["bm25_score"]
                hybrid_score = r[1]["hybrid_score"]

                print(f"{i+1}. {movieName}\nHybrid Score: {hybrid_score:.3f}\nBM25: {keyword_score:.3f}, Semantic: {semantic_score:.3f}\n{desc:100}")
        
        case "rrf-search":
            isEnhanced = args.enhance is not None
            reRankingEnabled = args.rerank_method is not None
            llm = LLM()

            query = args.query
            requestedLimit = args.limit
            modifiedLimit = requestedLimit

            if isEnhanced:
                query = llm.enhanceQuery(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            hrb = HybridSearch(loadMovies())
            if reRankingEnabled:
                modifiedLimit *= 5
            results = hrb.rrf_search(query, args.k, modifiedLimit)
            
            print(f"Re-ranking top {requestedLimit} results using individual method...\nReciprocal Rank Fusion Results for 'family movie about bears in the woods' (k={args.k}):")
            
            if args.rerank_method == "individual":
                for r in results:
                    r["rerank_score"] = llm.getIndividualReRankingScore(query, r)
            elif args.rerank_method == "batch":
                docListStr = ""
                for i, r in enumerate(results):
                    docListStr += f"\nindex={i} title={r.get('title', '')} description=({r.get('desc', '')})"
                
                rerankIndices = llm.getBatchReRankingScore(query, docListStr)
                # rerankedResults = []
                for i, index in enumerate(rerankIndices):
                    results[index]["rerank_score"] = len(rerankIndices) - i+1
            elif args.rerank_method == "cross_encoder":
                pairs = []
                for r in results:
                    pairs.append([query, f'{r.get("title", "")} - {r.get("desc", "")}'])
                encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = encoder.predict(pairs)
                for i in range(len(results)):
                    results[i]["rerank_score"] = scores[i]
                
            if reRankingEnabled:
                results.sort(key=lambda x: x["rerank_score"], reverse=True)

            for i, r in enumerate(results[:requestedLimit]):
                title = r["title"]
                desc = r["desc"][:100]
                rrf = r["rrf"]
                bm = r["bm25_score"]
                score = r["rerank_score"]
                semantic = r["semantic_score"]
                if args.rerank_method == "individual":
                    print(f"\n{i+1}. {title}\nRe-rank Score: {score}/10\nRRF Score: {rrf:.3f}\nBM25 Rank: {bm}, Semantic Rank: {semantic}\n{desc}")
                elif args.rerank_method == "batch":
                    print(f"\n{i+1}. {title}\nRe-rank Score: {len(results) - score + 2}\nRRF Score: {rrf:.3f}\nBM25 Rank: {bm}, Semantic Rank: {semantic}\n{desc}")
                elif args.rerank_method == "cross_encoder":
                    print(f"\n{i+1}. {title}\nCross Encoder Score: {score:.3f}\nRRF Score: {rrf:.3f}\nBM25 Rank: {bm}, Semantic Rank: {semantic}\n{desc}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()