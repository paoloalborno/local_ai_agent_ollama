import json
import sys
from tools import AgentTools
class Agent:

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.agent_tools = AgentTools(self.llm, self.retriever)

    def run_sequenced(self, user_query: str) -> str:
        """Execute tools in a fixed sequence and return JSON result"""
        try:
            print(f"Starting sequenced execution for: {user_query}", file=sys.stderr, flush=True)

            # Step 1: Extract keywords
            print("Step 1: Extracting keywords...", file=sys.stderr, flush=True)
            keywords = self.agent_tools.extract_important_keywords(user_query)
            print(f"Keywords extracted: {keywords}", file=sys.stderr, flush=True)

            # Step 2: Retrieve reviews
            print("Step 2: Retrieving reviews...", file=sys.stderr, flush=True)
            if isinstance(keywords, str):
                keywords_list = keywords.split(",")
            else:
                keywords_list = keywords
            reviews = self.agent_tools.retrieve_useful_reviews(keywords_list)
            print(f"Retrieved {len(reviews)} reviews", file=sys.stderr, flush=True)

            # Step 3: Summarize reviews
            print("Step 3: Summarizing reviews...", file=sys.stderr, flush=True)
            summary = self.agent_tools.summarize_reviews(reviews)
            print("Summary completed", file=sys.stderr, flush=True)

            # Step 4: Get statistics
            print("Step 4: Calculating statistics...", file=sys.stderr, flush=True)
            statistics = self.agent_tools.get_reviews_statistics(reviews)
            print("Statistics completed", file=sys.stderr, flush=True)

            # Step 5: Generate final JSON result
            result = {
                "query": user_query,
                "keywords": keywords_list,
                "reviews_count": len(reviews),
                "summary": summary,
                "statistics": statistics,
                "status": "success"
            }

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            error_result = {
                "query": user_query,
                "error": str(e),
                "status": "error"
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)

    def process_query(self, user_query: str) -> str:
        """Process query using sequenced execution"""
        return self.run_sequenced(user_query)

