import argparse
from lib.augment_generation import perform_rag, summarize_search, cited_summary, answer_search

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize = subparsers.add_parser("summarize", help="Summarize the search results")
    summarize.add_argument("query", type=str, help="Search for the query")
    summarize.add_argument("--limit", default=5, type=int, help="Limit the results used in summary")

    citation = subparsers.add_parser("citations", help="Summarize the search results with citations")
    citation.add_argument("query", type=str, help="Search for the query")
    citation.add_argument("--limit", default=5, type=int, help="Limit the results used in summary")

    question = subparsers.add_parser("question", help="Ask any movie questions within our dataset")
    question.add_argument("query", type=str, help="Search for the query")
    question.add_argument("--limit", default=5, type=int, help="Limit the results used in summary")

    args = parser.parse_args()

    query = args.query
    match args.command:
        case "rag":
            perform_rag(query, k=60, limit=5)
        case "summarize":
            summarize_search(query, k=60, limit=args.limit)
        case "citations":
            cited_summary(query, k=60, limit=args.limit)
        case "question":
            answer_search(query, k = 60, limit=args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()