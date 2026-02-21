#!/usr/bin/env python3
import argparse
import json
import helpers

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    
    match args.command:
        case "search":
            print('Searching for:', args.query)
            data = {}
            helpers.loadSaveWords()
            with open("data/movies.json") as f:
                data = json.load(fp=f)
                f.close()
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()