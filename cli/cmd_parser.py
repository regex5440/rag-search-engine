import argparse
import constants 

def attachParser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movie indexing for fast searches")

    tf = subparsers.add_parser("tf", help="Get term frequency in a document")
    tf.add_argument("docId", type=int, help="Id of the document")
    tf.add_argument("term", type=str, help="Term to be checked in document")
    
    idf = subparsers.add_parser("idf", help="Calculate the invert document frequency")
    idf.add_argument("term", type=str, help="Term to be searched across dataset")

    tfidf = subparsers.add_parser("tfidf", help="Get the best matching tf-idf")
    tfidf.add_argument("docId", type=int, help="Id of the document")
    tfidf.add_argument("term", type=str, help="Term to be checked in document")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=constants.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=constants.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="limit the search result, default 5")

    return parser

def semanticSearchParser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model information")

    embedding = subparsers.add_parser("embed_text", help="Get the embeddings of the provided text")
    embedding.add_argument("text", type=str, help="text value to generate embeddings for")

    subparsers.add_parser("verify_embeddings", help="Verify the movies embedding data")

    embedQ = subparsers.add_parser("embedquery", help="get the embeddings for provided query")
    embedQ.add_argument("query", type=str, help="string value to get the embedding vector for")

    search = subparsers.add_parser("search", help="Perform semantic search")
    search.add_argument("query", type=str, help="mandatory search query for semantic search")
    search.add_argument("--limit", type=int, default=5, help="Limit the search results, default 5")

    chunk = subparsers.add_parser("chunk", help="Chunk the text for optimized embeddings")
    chunk.add_argument("text", type=str, help="Text to be chunked")
    chunk.add_argument("--chunk-size", type=int, default=200, help="Define the chunk as word counts per chunk")
    chunk.add_argument("--overlap", type=int, default=0, help="Overlap the chunks")

    semantic_chunk = subparsers.add_parser("semantic_chunk" ,help="Perform semantic chunking for optimized embeddings")
    semantic_chunk.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="Maximum chunk size as number of words")
    semantic_chunk.add_argument("--overlap", default=0, type=int, help="Overlap the chunks by number of words")

    subparsers.add_parser("embed_chunks", help="Perform the semantic chunking of dataset")

    search = subparsers.add_parser("search_chunked", help="Perform the search based on semantic embeddings")
    search.add_argument("query", type=str)
    search.add_argument("--limit", type=int, default=5)

    return parser

def hybridSearchParser():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sp = parser.add_subparsers(dest="command", help="Available commands")
    nrmlz = sp.add_parser("normalize", help="Normalize the BM25 score list to Semantic's Cosine level")
    nrmlz.add_argument("list", nargs="+", type=float, help="sequence of numbers separated with whitespace")

    ws = sp.add_parser("weighted-search", help="Search based on hybrid search")
    ws.add_argument("query", type=str, help="Mandatory query")
    ws.add_argument("--alpha", type=float, default=0.5, help="Set the weight of keyword/semantic search")
    ws.add_argument("--limit", type=int, default=5, help="Limit the search results")

    rrfsearch = sp.add_parser("rrf-search", help="Perform Reciprocal Rank Fusion")
    rrfsearch.add_argument("query", type=str, help="Mandatory query to search against")
    rrfsearch.add_argument("-k", default=60, type=int, help="Set the K parameter")
    rrfsearch.add_argument("--limit", default=5, type=int, help="Limit the search results")
    rrfsearch.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Enhance user query with LLM")
    rrfsearch.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Enhance results with LLM based re-ranking")
    rrfsearch.add_argument("--evaluate", action="store_true", help="Manually evaluate the RRF results")

    return parser


def evaluationParser():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    return parser