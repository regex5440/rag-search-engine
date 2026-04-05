from os import environ
from google import genai
from dotenv import load_dotenv
import json

class LLM:
    def __init__(self):
        load_dotenv()
        GEMINI_API_KEY = environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.MODEL = "gemma-3-27b-it"

    def enhanceQuery(self, query: str, command: str) -> str:
        prompt = ""
        match command:
            case "spell":
                prompt = self.__withSpellCheck(query)
            case "rewrite":
                prompt = self.__withRewrite(query)
            case "expand":
                prompt = self.__withExtendedContext(query)
            case "":
                raise Exception("Unhandled enhance command")

        response = self.client.models.generate_content(model=self.MODEL,contents=prompt)
        if response.text is None:
            return query
        if command == "expand":
            return response.text + " " + query
        return response.text
    
    def getIndividualReRankingScore(self, query, doc) -> int:

        prompt = f"""Rate how well this movie matches the search query.
Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("desc", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        response = self.client.models.generate_content(model=self.MODEL,contents=prompt)
        if response.text is None:
            return 0
        return int(response.text)
    
    def getBatchReRankingScore(self, query, docListStr) -> list:
        prompt = prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{docListStr}

Return ONLY the indices in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 0, 12, 1]

Ranking:"""
        response = self.client.models.generate_content(model=self.MODEL,contents=prompt)
        if response.text is None:
            return []
        return json.loads(response.text)
    
    def evaluationScores(self, query, formatted_results):
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
        response = self.client.models.generate_content(model=self.MODEL, contents=prompt)
        if response.text is None:
            return []
        return json.loads(response.text)

    def __withSpellCheck(self, q: str)->str:
        return f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{q}"
"""
    def __withRewrite(self, q: str) -> str:
        return f"""Rewrite the user-provided movie search query below to be more specific and searchable.
Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{q}"
"""
    
    def __withExtendedContext(self, q: str) -> str:
        return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{q}"
"""
