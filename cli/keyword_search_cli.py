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
    idx = InvertedIndex()
    data = {}
    with open("data/movies.json") as f:
        data = json.load(fp=f)
        f.close()
    match args.command:
        case "search":
            qTokens = helpers.tokenizeSearchTerm(args.query)
            try:
                idx.load()
            except FileNotFoundError:
                print("indexing does not exists")
                return
            print('Searching for:', args.query)
            matchedIds = set()
            for token in qTokens:
                enough = False
                for r in idx.get_documents(token):
                    if len(matchedIds) >= 5:
                        enough = True
                        break
                    matchedIds.add(r)
                if enough:
                    break
            for id in matchedIds:
                movie = idx.docmap[id]
                print(f'{id} {movie["title"]}')
        case "build":
            idx.build(data["movies"])
            idx.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()