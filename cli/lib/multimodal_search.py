from typing import Any, cast
from .semantic_search import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer
from helpers import loadMovies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", docs=[]):
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = []
        for doc in docs:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    def embed_image(self, imageFile):
        with Image.open(imageFile) as image:
            embeddings = cast(Any, self.model).encode([image], show_progress_bar=True)
            return embeddings[0]
    
    def search_with_image(self, img_path):
        imgembd = self.embed_image(img_path)
        for i, txt_embd in enumerate(self.text_embeddings):
            self.docs[i]["score"] = cosine_similarity(imgembd, txt_embd)
        
        return sorted(self.docs, key=lambda x: x["score"], reverse=True)[:5]
    
def image_search_command(image_path):
    mms = MultimodalSearch(docs = loadMovies())
    return mms.search_with_image(image_path)
        
def verify_image_embedding(image_path):
    mms = MultimodalSearch()
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
