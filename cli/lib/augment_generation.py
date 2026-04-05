from helpers import loadMovies
from .hybrid_search import HybridSearch
from .llm import LLM
import json

def perform__rrf(query, k ,limit):
    movies = loadMovies()
    hybrid = HybridSearch(movies)
    results = hybrid.rrf_search(query,limit=limit, k=k)
    print("Search Results:")
    formatted_results = []
    for r in results:
        print(f'- {r.get("title", "")}')
        formatted_results.append(f'{r.get("title","")} - {r.get("desc", "")}')
    return formatted_results

def perform_rag(query, k, limit):
    results = perform__rrf(query, k, limit)
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{json.dumps(results)}

Answer:"""
    
    llm = LLM()
    response = llm.raw_generation(prompt)
    if response.text is not None:
        print("\nRAG Response:")
        print(response.text)
        return
    print("Unable to generate RAG response")

def summarize_search(query, k, limit):
    results = perform__rrf(query, k, limit)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""
    
    llm = LLM()
    response = llm.raw_generation(prompt)
    if response.text is not None:
        print("\nLLM Summary:")
        print(response.text)
        return
    print("Unable to generate RAG response")

def cited_summary(query, k, limit):
    results = perform__rrf(query, k, limit)
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    llm = LLM()
    response = llm.raw_generation(prompt)
    if response.text is not None:
        print("\nLLM Answer:")
        print(response.text)
        return
    print("Unable to generate RAG response")

def answer_search(query, k, limit):
    results = perform__rrf(query, k, limit)
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {query}

Documents:
{results}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    llm = LLM()
    response = llm.raw_generation(prompt)
    if response.text is not None:
        print("\nAnswer:")
        print(response.text)
        return
    print("Unable to generate RAG response")