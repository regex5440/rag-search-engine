#!/usr/bin/env python3
from cmd_parser import semanticSearchParser
from lib import semantic_search
from lib import chunked_semantic_search

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
        case "chunk":
            chunkSize = args.chunk_size
            charLength = len(args.text)
            textSplit = args.text.split()
            overlap = args.overlap
            i = 0
            chunks = []
            while i < len(textSplit):
                start = min(i, abs(i-overlap))
                end = min((i + chunkSize), len(textSplit))
                chunks.append(" ".join(textSplit[start:end]))
                i = end
            
            print(f"Chunking {charLength} characters")
            for t in range(len(chunks)):
                print(f"{t+1}. {chunks[t]}")
        case "semantic_chunk":
            chunkSize = args.max_chunk_size
            overlap = args.overlap
            chunks = chunked_semantic_search.semantic_chunking(args.text, chunkSize, overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks):
                print(f"{i+1}. {c}")

        case "embed_chunks":
            embeddings = chunked_semantic_search.get_semantic_chunk_embedding()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            results = chunked_semantic_search.semantic_search(args.query, args.limit)
            if results is None:
                print("no results!")
                return
            for i, movie in enumerate(results):
                TITLE = movie["title"]
                SCORE = movie["score"]
                DOCUMENT = movie["document"]
                print(f"\n{i+1}. {TITLE} (score: {SCORE:.4f})")
                print(f"   {DOCUMENT}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()