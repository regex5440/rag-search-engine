from lib import semantic_search as sse
import re
import numpy as np
import json
from constants import SEMANTIC_MOVIE_EMBEDDINGS_FILE, CACHE_FOLDER, SEMANTIC_CHUNK_METADATA_FILE, MOVIES_DATA_FILE
from os import makedirs, path

class ChunkedSemanticSearch(sse.SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list):
        self.documents = documents

        allChunks: list[str] = []
        self.chunk_metadata = []
        print("Generating document embeddings...")
        for i, doc in enumerate(self.documents):
            id = doc["id"]
            self.document_map[id] = doc

            desc = doc["description"]

            if len(desc) == 0:
                continue
            chunks = semantic_chunking(desc, 4, 1)
            chunkLen = len(chunks)
            for ci, c in enumerate(chunks):
                allChunks.append(c)
                self.chunk_metadata.append({
                    "movie_idx": i,
                    "chunk_idx": ci,
                    "total_chunks": chunkLen
                })
        self.chunk_embeddings = self.model.encode(allChunks, show_progress_bar=True)

        makedirs(CACHE_FOLDER, exist_ok=True)
        with open(SEMANTIC_MOVIE_EMBEDDINGS_FILE, "wb") as f:
            np.save(f, self.chunk_embeddings)
            print(f"Saved chunk embeddings to {SEMANTIC_MOVIE_EMBEDDINGS_FILE}...")
            f.close()
        with open(SEMANTIC_CHUNK_METADATA_FILE, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(allChunks)}, f, indent=2)
            print(f"Saved chunk metadata to {SEMANTIC_CHUNK_METADATA_FILE}...")
            f.close()
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if not path.exists(SEMANTIC_MOVIE_EMBEDDINGS_FILE) or not path.exists(SEMANTIC_CHUNK_METADATA_FILE):
            print("Cache miss! Generating new embeddings")
            return self.build_chunk_embeddings(documents)
        self.documents = documents
        with open(SEMANTIC_MOVIE_EMBEDDINGS_FILE, "rb") as f:
            self.chunk_embeddings = np.load(f)
            f.close()
        with open(SEMANTIC_CHUNK_METADATA_FILE) as f:
            data = json.load(f)
            self.chunk_metadata = data["chunks"]
            f.close()
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        qEmb = self.generate_embedding(query)
        chunkScore = []
        movieScoreMap = {}
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            print("chunk cache embedding or metadata not loaded!")
            return
        for i, chEmb in enumerate(self.chunk_embeddings):
            score = sse.cosine_similarity(chEmb, qEmb)
            metadata = self.chunk_metadata[i]
            movie_idx = metadata["movie_idx"]
            chunkScore.append({
                "chunk_idx": i,
                "movie_idx": movie_idx,
                "score": score
            })
            if movie_idx not in movieScoreMap or movieScoreMap[movie_idx] < score:
                movieScoreMap[movie_idx] = score

        results = []
        for [idx, score] in sorted(movieScoreMap.items(), reverse=True, key=lambda x: x[1])[:limit]:
            movie = self.documents[idx]
            results.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(score,2),
                "metadata": {}
            })
        return results


def semantic_chunking(text: str, chunkSize: int, overlap: int):
    text = text.strip()
    if text == "":
        return []
    textSplit = re.split(r"(?<=[.!?])\s+",text)
    chunks = []
    i = 0
    while i < len(textSplit):
        start = min(i, abs(i - overlap))
        stop = min(start + chunkSize, len(textSplit))
        chunk = (" ".join(textSplit[start:stop])).strip()
        if chunk == "":
            continue
        chunks.append(chunk)
        i = stop
    return chunks

def get_semantic_chunk_embedding():
    cse = ChunkedSemanticSearch()
    with open(MOVIES_DATA_FILE) as f:
        documents = json.load(f)
        f.close()
        movies = documents["movies"]
        return cse.load_or_create_chunk_embeddings(movies)

def semantic_search(query: str, limit: int):
    cse = ChunkedSemanticSearch()
    with open(MOVIES_DATA_FILE) as f:
        documents = json.load(f)
        f.close()
        movies = documents["movies"]
        cse.load_or_create_chunk_embeddings(movies)
    return cse.search_chunks(query, limit)
    