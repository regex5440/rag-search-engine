from sentence_transformers import SentenceTransformer
import numpy as np
from constants import CACHE_FOLDER, MOVIE_EMBEDDINGS_FILE, MOVIES_DATA_FILE
from os import makedirs, path
import json

class SemanticSearch:
    def __init__(self, model = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        sanitized_text = text.strip()
        if sanitized_text == "":
            raise ValueError("provided empty text")
        result = self.model.encode([sanitized_text])
        return result[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_text = []
        for doc in self.documents:
            id = doc["id"]
            title = doc["title"]
            desc = doc["description"]
            self.document_map[id] = doc
            doc_text.append(f"{title}:{desc}")
        self.embeddings = self.model.encode(doc_text, show_progress_bar=True)
        self.__save()
        return self.embeddings

    def __save(self):
        if self.embeddings is None:
            print("no embeddings to save")
            return
        makedirs(CACHE_FOLDER, exist_ok=True)
        with open(MOVIE_EMBEDDINGS_FILE, "wb") as f:
            np.save(f, self.embeddings)
            f.close()
    
    def load_or_create_embeddings(self, documents):
        if self.embeddings != None and len(self.embeddings) == len(documents):
            return self.embeddings
        self.documents = documents
        if path.exists(MOVIE_EMBEDDINGS_FILE):
            with open(MOVIE_EMBEDDINGS_FILE, "rb") as f:
                self.embeddings = np.load(f)
                f.close()
                if len(self.embeddings) == len(documents):
                    return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qEmb = self.generate_embedding(query)
        matchingScore = []
        for i in range(len(self.documents)):
            matchingScore.append((cosine_similarity(self.embeddings[i], qEmb),self.documents[i]))
        results = []
        for (score, doc) in sorted(matchingScore, reverse=True, key= lambda x: x[0])[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return results



def embed_text(text):
    sse = SemanticSearch()
    embedding = sse.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    sse = SemanticSearch()
    print(f"Model loaded: {sse.model}\nMax sequence length: {sse.model.max_seq_length}")

def verify_embeddings():
    sse = SemanticSearch()
    with open(MOVIES_DATA_FILE) as f:
        documents = json.load(f)
        movies = documents["movies"]
        embeddings = sse.load_or_create_embeddings(movies)
        print(f"Number of docs:   {len(movies)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query):
    sse = SemanticSearch()
    embedding = sse.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product/(norm1 * norm2)

def perform_semantic_search_cmd(query: str, limit: int):
    sse = SemanticSearch()
    with open(MOVIES_DATA_FILE) as f:
        data = json.load(f)
        f.close()
        movies = data["movies"]
        sse.load_or_create_embeddings(movies)
    return sse.search(query, limit)