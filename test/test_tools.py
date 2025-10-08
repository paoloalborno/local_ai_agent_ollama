import pytest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DummyLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        # Simulate LLM output based on prompt content
        if "extract" in prompt.lower():
            return "mouse,wireless,rgb,battery,dpi"
        if "summarize" in prompt.lower():
            return (
                "Summary: The reviews highlight that the mouse is highly responsive and comfortable for long gaming sessions. "
                "Many users appreciate the RGB lighting and customizable DPI settings. "
                "Some mention battery life could be improved, but overall sentiment is positive."
            )
        if "analyze" in prompt.lower() or "compute" in prompt.lower():
            return (
                "Average rating: 4.1, Range: 2-5, Total: 10, Positive: 7, Negative: 2"
            )
        return "test"


class DummyDoc:
    def __init__(self, content, rating, date, title):
        self.page_content = content
        self.metadata = {"rating": rating, "date": date, "title": title}


class DummyVectorStore:
    def similarity_search_with_score(self, query, k=5):
        # Restituisce tuple (documento, score) dove score è la distanza
        docs = [
            (DummyDoc("Great mouse, very responsive and ergonomic.", 5, "2024-01-01", "Awesome Mouse"), 0.1),
            (DummyDoc("Not bad, but battery life is short.", 3, "2024-01-02", "Okay Mouse"), 0.3),
            (DummyDoc("Excellent RGB lighting and smooth tracking.", 5, "2024-01-03", "RGB Mouse"), 0.2),
            (DummyDoc("Stopped working after a month.", 2, "2024-01-04", "Disappointing"), 0.8),
            (DummyDoc("Very lightweight and perfect for FPS games.", 4, "2024-01-05", "FPS Mouse"), 0.4),
            (DummyDoc("Bluetooth connection sometimes drops.", 3, "2024-01-06", "Bluetooth Issue"), 0.6),
            (DummyDoc("Customizable macros are a game changer.", 5, "2024-01-07", "Macro Mouse"), 0.25),
            (DummyDoc("DPI settings are easy to adjust.", 4, "2024-01-08", "DPI Mouse"), 0.3),
            (DummyDoc("Battery lasts for days, very happy.", 5, "2024-01-09", "Long Battery"), 0.35),
            (DummyDoc("A bit heavy for my taste, but works well.", 3, "2024-01-10", "Heavy Mouse"), 0.5),
        ]
        return docs[:k]


class DummyRetriever:
    def __init__(self):
        self.vectorstore = DummyVectorStore()


from tools import AgentTools


@pytest.fixture
def agent_tools():
    llm = DummyLLM()
    retriever = DummyRetriever()
    return AgentTools(llm, retriever)


def test_extract_important_keywords(agent_tools):
    keywords = agent_tools.extract_important_keywords("I want a wireless RGB mouse with good battery and macros")
    print("\n[TEST] extract_important_keywords prompt:\n", agent_tools.llm.last_prompt)
    print("[TEST] extract_important_keywords result:", keywords)
    assert isinstance(keywords, list)
    assert len(keywords) == 5
    keywords_lower = [k.lower() for k in keywords]
    assert "mouse" in keywords_lower


def test_retrieve_useful_reviews(agent_tools):
    reviews = agent_tools.retrieve_useful_reviews(["mouse", "wireless"], k=5, min_similarity=0.5)
    print("\n[TEST] retrieve_useful_reviews result (first 2):", reviews[:2])
    assert isinstance(reviews, list)
    assert len(reviews) >= 2
    for review in reviews:
        assert "content" in review
        assert "similarity" in review
        assert review["similarity"] >= 0.5


def test_similarity_calculation(agent_tools):
    reviews = agent_tools.retrieve_useful_reviews(["mouse"], k=3, min_similarity=0.0)
    print("\n[TEST] similarity values:", [r["similarity"] for r in reviews])

    assert reviews[0]["similarity"] == 0.9


def test_summarize_reviews(agent_tools):
    reviews = [
        {"title": "Awesome Mouse", "rating": 5, "date": "2024-01-01",
         "content": "Great mouse, very responsive and ergonomic."},
        {"title": "Okay Mouse", "rating": 3, "date": "2024-01-02",
         "content": "Not bad, but the battery life is quite short."},
    ]
    summary = agent_tools.summarize_reviews(reviews)
    print("\n[TEST] summarize_reviews result:", summary)
    assert isinstance(summary, str)
    assert "Summary" in summary
    assert len(summary) > 50


def test_get_reviews_statistics(agent_tools):
    reviews = [
        {"content": "Great mouse", "rating": 5, "date": "2024-01-01", "title": "Awesome Mouse"},
        {"content": "Not bad", "rating": 3, "date": "2024-01-02", "title": "Okay Mouse"},
        {"content": "Excellent RGB", "rating": 5, "date": "2024-01-03", "title": "RGB Mouse"},
    ]
    stats = agent_tools.get_reviews_statistics(reviews)
    print("\n[TEST] get_reviews_statistics result:", stats)
    assert isinstance(stats, str)
    assert "Average rating" in stats or "average" in stats.lower()


def test_integration_workflow(agent_tools):
    """Test the complete workflow: extract -> retrieve -> summarize -> statistics"""
    print("\n[TEST] Testing complete workflow...")

    user_query = "I want a wireless gaming mouse with RGB lighting"
    keywords = agent_tools.extract_important_keywords(user_query)
    print(f"Step 1 - Keywords: {keywords}")
    assert len(keywords) == 5

    reviews = agent_tools.retrieve_useful_reviews(keywords[:3], k=5, min_similarity=0.1)
    print(f"Step 2 - Retrieved {len(reviews)} reviews")
    assert len(reviews) >= 3

    summary = agent_tools.summarize_reviews(reviews)
    print(f"Step 3 - Summary length: {len(summary)} chars")
    assert len(summary) > 50

    stats = agent_tools.get_reviews_statistics(reviews)
    print(f"Step 4 - Stats: {stats[:100]}...")
    assert "Average" in stats or "average" in stats.lower()

    print("[TEST] Complete workflow successful!")
