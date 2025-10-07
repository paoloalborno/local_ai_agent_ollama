from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaLLM

class AgentTools:
    def __init__(self, llm: OllamaLLM, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

        self.prompt_keywords = (
            "Given the following user query your goal is to extract the most important keywords on order to retrieve related reviews via RAG.\n"
            "Add only very close synonyms. "
            "User query: {user_query}\n"
            "Return ONLY a comma-separated list of keywords, no explanations."
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
            return keywords[:5] if keywords else [user_query]
        except Exception as e:
            return [{"error": f"Extraction failed: {str(e)}"}]

    def retrieve_useful_reviews(self, keywords: List[str], k: int = 5) -> List[Dict[str, Any]]:
        try:
            search_query = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
            docs = self.retriever.invoke(search_query)
            results = []
            for doc in docs[:k]:
                results.append({
                    "content": doc.page_content,
                    "rating": doc.metadata.get("rating"),
                    "date": doc.metadata.get("date"),
                    "title": doc.metadata.get("title")
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
