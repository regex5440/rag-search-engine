from pickle import dump, load
import os
from helpers import tokenizeSearchTerm

class InvertedIndex():
    def __init__(self) -> None:
        self.index = {} # tokens: str -> set[int] of docIds
        self.docmap = {} # docId: int -> docObject
        
        self.__cacheLocation = os.path.join(os.getcwd(),"cache")
        self.__indexFile = os.path.join(self.__cacheLocation, "index.pkl")
        self.__docMapFile = os.path.join(self.__cacheLocation, "docmap.pkl")

    def __add_document(self, doc_id, text):
        for token in tokenizeSearchTerm(text):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str):
        [token] = tokenizeSearchTerm(term)
        if token is None:
            return []
        if token not in self.index:
            return []
        
        return sorted(self.index[token])
    
    def build(self, moviesList):
        i = 1
        for m in moviesList:
            self.docmap[i] = m
            self.__add_document(i, f"{m['title']} {m['description']}")
            i +=1

    def save(self):
        print("Saving Indexed data...")
        os.makedirs(self.__cacheLocation, exist_ok=True)

        with open(self.__indexFile, "wb") as f:
            dump(self.index, f)
            f.close()
            print("Saved index to", self.__indexFile)
        with open(self.__docMapFile, "wb") as f:
            dump(self.docmap, f)
            f.close()
            print("Saved movie mapping to", self.__docMapFile)

    def load(self):
        if len(self.index) > 0 and len(self.docmap) > 0:
            return
        if not os.path.exists(self.__cacheLocation):
            raise FileNotFoundError()
        with open(self.__indexFile, "rb") as f:
            self.index = load(f)
            f.close()
        with open(self.__docMapFile, "rb") as f:
            self.docmap = load(f)
            f.close()