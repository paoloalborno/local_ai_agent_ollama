from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaLLM

class AgentTools:
    def __init__(self, llm: OllamaLLM, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

        self.prompt_keywords = (
            "The text below is a user query. Extract only the key terms needed to retrieve related reviews via RAG. "
            "If any, include in keyword list ONLY -same meaning synonyms-"
            "Do not add explanations, categories, or commentary. "
            "Output format must be exactly: k1,k2,k3,...,kn — a single comma-separated line with no spaces, "
            "no quotes, no text before or after.\n"
            "User query: {user_query}"
        )

        self.prompt_summary = (
            "Summarize the following reviews. Focus on:\n"
            "- Main pros and cons\n"
            "- Recurring themes\n"
            "- Overall sentiment\n"
            "- Key recommendations\n\n"
            "Reviews: {reviews}\n\n"
            "Provide a concise summary in 2-3 paragraphs."
        )
        self.prompt_stats = (
            "Analyze the following reviews and compute:\n"
            "- Average rating\n"
            "- Rating range\n"
            "- Total number of reviews\n"
            "- How many are positive (4-5) and how many are negative (1-2)\n\n"
            "Reviews: {reviews}\n\n"
            "Reply concisely."
        )

    def extract_important_keywords(self, user_query: str) -> List[str]:
        prompt = self.prompt_keywords.format(user_query=user_query)
        try:
            response = self.llm.invoke(prompt)
            keywords = [k.strip() for k in response.split(',') if k.strip()]
            return keywords[:5]
        except Exception as e:
            return [{"error": f"Extraction failed: {str(e)}"}]

    def retrieve_useful_reviews(self, keywords: List[str], k: int = 5, min_similarity: float = 0.15) -> List[Dict[str, Any]]:
        try:
            search_query = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
            docs_with_scores = self.retriever.vectorstore.similarity_search_with_score(search_query, k=k)
            results = []
            for doc, score in docs_with_scores:
                similarity = 1 - score  # distance is converted (score is - cosine_similarity = (A · B) / (||A|| * ||B||)

                # only the most similar documents are returned
                if similarity >= min_similarity:
                    results.append({
                        "content": doc.page_content,
                        "rating": doc.metadata.get("rating"),
                        "date": doc.metadata.get("date"),
                        "title": doc.metadata.get("title"),
                        "similarity": similarity
                    })
            return results
        except Exception as e:
            return [{"error": f"Retrieval failed: {str(e)}"}]

    def summarize_reviews(self, reviews: list) -> str:
        try:
            prompt = self.prompt_summary.format(reviews=reviews)
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def get_reviews_statistics(self, reviews: list) -> str:
        try:
            ratings = []
            total_reviews = 0
            for review in reviews:
                total_reviews += 1
                if isinstance(review, dict) and review.get("rating"):
                    try:
                        ratings.append(float(review["rating"]))
                    except Exception:
                        pass

            prompt = self.prompt_stats.format(reviews=reviews)
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Statistics calculation failed: {str(e)}"
