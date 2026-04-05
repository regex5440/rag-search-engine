from json import load
from cmd_parser import evaluationParser
from constants import GOLDEN_DATASET
from lib.hybrid_search import HybridSearch
from helpers import loadMovies

def main():
    parser = evaluationParser()
    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET) as f:
        golden_data = load(f)
        search = HybridSearch(loadMovies())
        K = 60
        print(f"k={K}\n")
        for t_case in golden_data["test_cases"]:
            query = t_case.get("query","")
            relevant = t_case.get("relevant_docs", [])
            retrieved: list[str] = [r.get("title", "") for r in search.rrf_search(query, K, limit)[:limit]]
            relevant_retrieved_count = 0
            for title in retrieved:
                if title in relevant:
                    relevant_retrieved_count+=1
            
            actual_retrieved_count = len(retrieved)
            precision = relevant_retrieved_count/actual_retrieved_count if actual_retrieved_count > 0 else 0
            recall = relevant_retrieved_count/len(relevant) if len(relevant) > 0 else 0
            
            if precision + recall == 0:
                f1score = 0.0
            else:
                f1score = 2 * (precision * recall) / (precision + recall)
            
            print(f"\n- Query: {query}\n\t- Precision@{limit}: {precision:.4f}\n\t- Recall@{limit}: {recall:0.4f}\n\t- F1 Score: {f1score:.4f}\n\t- Retrieved ({actual_retrieved_count}): {', '.join(retrieved)}\n\t- Relevant ({len(relevant)}): {', '.join(relevant)}")


if __name__ == "__main__":
    main()