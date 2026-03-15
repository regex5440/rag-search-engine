#!/usr/bin/env python3
from cmd_parser import semanticSearchParser
from lib import semantic_search

def main():
    parser = semanticSearchParser()
    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embedquery":
            semantic_search.embed_query_text(args.query)
        case "search":
            results = semantic_search.perform_semantic_search_cmd(args.query, args.limit)
            for i in range(len(results)):
                title = results[i]["title"]
                score = results[i]["score"]
                description = results[i]["description"]

                print(f"{i+1}. {title} (score: {score})\n\t{description:70}\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()