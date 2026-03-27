from cmd_parser import hybridSearchParser
from helpers import loadMovies
from lib.hybrid_search import HybridSearch

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
            hrb = HybridSearch(loadMovies())
            results = hrb.rrf_search(args.query, args.k, args.limit)
            for i, r in enumerate(results):
                title = r["title"]
                desc = r["desc"]
                rrf = r["rrf"]
                bm = r["bm25_score"]
                semantic = r["semantic_score"]
                print(f"{i+1}. {title}\nRRF Score: {rrf:.3f}\nBM25 Rank: {bm}, Semantic Rank: {semantic}\n{desc}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()