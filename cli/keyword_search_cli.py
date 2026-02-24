#!/usr/bin/env python3
import argparse
import json
import helpers
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build movie indexing for fast searches")
    

    args = parser.parse_args()

    
    helpers.loadSaveWords()
    data = {}
    with open("data/movies.json") as f:
        data = json.load(fp=f)
        f.close()
    match args.command:
        case "search":
            print('Searching for:', args.query)
            count = 0

            qTokens = helpers.tokenizeSearchTerm(args.query)
            # Query matching logic
            for movies in data["movies"]:
                if count >=5:
                    break
                movieTitleTokens = helpers.tokenizeSearchTerm(movies["title"])
                        
                for token in qTokens:
                    matchFound = False
                    for movieToken in movieTitleTokens:
                        if token in movieToken:
                            matchFound = True
                            break
                    if matchFound:
                        count+=1
                        print(f'{count}. {movies["title"]}')
                        # print(movieTitleTokens)
                        break
        case "build":
            idx = InvertedIndex()
            idx.build(data["movies"])
            idx.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()