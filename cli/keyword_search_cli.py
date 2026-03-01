#!/usr/bin/env python3
import argparse
import json
import helpers
from inverted_index import InvertedIndex
from math import log

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build movie indexing for fast searches")

    tf = subparsers.add_parser("tf", help="Get term frequency in a document")
    tf.add_argument("docId", type=int, help="Id of the document")
    tf.add_argument("term", type=str, help="Term to be checked in document")
    
    idf = subparsers.add_parser("idf", help="Calculate the invert document frequency")
    idf.add_argument("term", type=str, help="Term to be searched across dataset")

    tfidf = subparsers.add_parser("tfidf", help="Get the best matching tf-idf")
    tfidf.add_argument("docId", type=int, help="Id of the document")
    tfidf.add_argument("term", type=str, help="Term to be checked in document")

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
        
        case "tf":
            idx.load()
            print("Frequency",args.docId, args.term,idx.get_tf(args.docId, args.term))
        case "idf":
            idx.load()
            matchingDocCount = len(idx.get_documents(args.term))
            idf = log((len(idx.docmap)+1)/(matchingDocCount+1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            idx.load()
            tf = idx.get_tf(args.docId, args.term)
            matchingDocsCount = len(idx.get_documents(args.term))
            idf = log((len(idx.docmap) + 1)/(matchingDocsCount+1))
            tf_idf = tf*idf
            print(f"TF-IDF score of '{args.term}' in document '{args.docId}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()