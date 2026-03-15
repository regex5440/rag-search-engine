from os import path, getcwd
CACHE_FOLDER = path.join(getcwd(),"cache")
INDEX_FILE = path.join( CACHE_FOLDER,"index.pkl")
DOCUMENT_OBJ_FILE = path.join(CACHE_FOLDER, "docmap.pkl")
TERM_FREQUENCY_FILE = path.join(CACHE_FOLDER, "term_frequencies.pkl")
DOC_LENGTH_FILE = path.join(CACHE_FOLDER, "doc_lengths.pkl")
MOVIE_EMBEDDINGS_FILE = path.join(CACHE_FOLDER, "movie_embeddings.npy")

MOVIES_DATA_FILE = path.join("data", "movies.json")


BM25_K1 = 1.5
BM25_B = 0.75

