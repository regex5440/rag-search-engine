from pickle import dump, load
import os
from helpers import tokenizeSearchTerm
from collections import Counter

class InvertedIndex():
    def __init__(self) -> None:
        self.index = {} # tokens: str -> set[int] of docIds
        self.docmap = {} # docId: int -> docObject
        self.term_frequency: dict[int, Counter] = {} # docId -> Counter
        
        self.__cacheLocation = os.path.join(os.getcwd(),"cache")
        self.__indexFile = os.path.join(self.__cacheLocation, "index.pkl")
        self.__docMapFile = os.path.join(self.__cacheLocation, "docmap.pkl")
        self.__termFrequency = os.path.join(self.__cacheLocation, "term_frequencies.pkl")

    def __add_document(self, doc_id, text):
        for token in tokenizeSearchTerm(text):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id].update([token])

    def get_tf(self, doc_id, term) -> int:
        tokens = tokenizeSearchTerm(term)
        if len(tokens) > 1:
            raise Exception("only singly word is expected")
        if doc_id not in self.term_frequency:
            return 0
        return self.term_frequency[doc_id].get(tokens[0]) or 0

    def get_documents(self, term: str):
        [token] = tokenizeSearchTerm(term)
        if token is None:
            return []
        if token not in self.index:
            return []
        
        return sorted(self.index[token])
    
    def build(self, moviesList):
        for m in moviesList:
            id = m["id"]
            self.docmap[id] = m
            self.__add_document(id, f"{m['title']} {m['description']}")

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
        with open(self.__termFrequency, "wb") as f:
            dump(self.term_frequency, f)
            f.close()
            print("Saved term frequencies to", self.__termFrequency)

    def load(self):
        if len(self.index) > 0 and len(self.docmap) > 0 and len(self.term_frequency) > 0:
            return
        if not os.path.exists(self.__cacheLocation):
            raise FileNotFoundError()
        with open(self.__indexFile, "rb") as f:
            self.index = load(f)
            f.close()
        with open(self.__docMapFile, "rb") as f:
            self.docmap = load(f)
            f.close()
        with open(self.__termFrequency, "rb") as f:
            self.term_frequency = load(f)
            f.close()
