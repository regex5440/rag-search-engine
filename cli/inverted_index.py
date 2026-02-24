from pickle import dump
import os
from helpers import tokenizeSearchTerm

class InvertedIndex():
    def __init__(self) -> None:
        self.index = {} # tokens: str -> set[int] of docIds
        self.docmap = {} # docId: int -> docObject

    def __add_document(self, doc_id, text):
        for token in tokenizeSearchTerm(text):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str):
        [token] = tokenizeSearchTerm(term)
        if token is None:
            return []
        if token not in self.index[token]:
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
        cacheLocation = os.path.join(os.getcwd(),"cache")
        os.makedirs(cacheLocation, exist_ok=True)

        indexFile = os.path.join(cacheLocation, "index.pkl")
        docMapFile = os.path.join(cacheLocation, "docmap.pkl")

        with open(indexFile, "wb") as f:
            dump(self.index, f)
            f.close()
            print("Saved index to", indexFile)
        with open(docMapFile, "wb") as f:
            dump(self.docmap, f)
            f.close()
            print("Saved movie mapping to", docMapFile)

