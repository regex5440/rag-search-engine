#!/usr/bin/env python3
import helpers
from cli.lib.inverted_index import InvertedIndex
from math import log
import cmd_parser

def main() -> None:
    parser = cmd_parser.attachParser()
    args = parser.parse_args()
    
    helpers.loadSaveWords()
    idx = InvertedIndex()

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
            idx.build()
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
        case "bm25idf":
            idx.load()
            bm25idf = idx.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            idx.load()
            bm25tf = idx.get_bm25_tf(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            idx.load()
            result = idx.bm25_search(args.query, args.limit)
            for i, [id, score] in enumerate(result):
                movie = idx.docmap.get(id,{})
                title = movie['title']
                print(f"{i+1}. ({id}) {title} - Score: {score:.2f}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()