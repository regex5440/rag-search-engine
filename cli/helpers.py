from nltk.stem import PorterStemmer

translationTable = str.maketrans({
                "!": None,
                ",": None,
                "'": None,
                "\"": None,
            })

stopWords = []

def loadSaveWords():
    global stopWords
    if len(stopWords) > 0:
        return
    with open("data/stopwords.txt") as f:
        stopWords = f.read().splitlines()


def tokenizeSearchTerm(q):
    # 1. Prevent Case-Sensitivity
    sensitiveQuery = str(q).lower()

    # 2. Remove any punctuation
    sensitiveQuery.translate(translationTable) 

    # 3. Tokenization
    __tokenizedQuery = sensitiveQuery.split(" ")
    queryTokens = []    
    for word in __tokenizedQuery:
        if len(word) > 0 and word not in stopWords:   #4. Stop words
            queryTokens.append(word)

    # 5. Stemming
    stemmer = PorterStemmer()

    for i in range(len(queryTokens)):
        queryTokens[i] = stemmer.stem(queryTokens[i])

    return queryTokens