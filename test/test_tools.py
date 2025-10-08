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
            return "mouse, wireless, rgb, battery, dpi, gaming, ergonomic, bluetooth, lightweight, macro"
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

class DummyRetriever:
    def invoke(self, query):
        class DummyDoc:
            def __init__(self, content, rating, date, title):
                self.page_content = content
                self.metadata = {"rating": rating, "date": date, "title": title}
        # 10 varied reviews
        return [
            DummyDoc("Great mouse, very responsive and ergonomic.", 5, "2024-01-01", "Awesome Mouse"),
            DummyDoc("Not bad, but battery life is short.", 3, "2024-01-02", "Okay Mouse"),
            DummyDoc("Excellent RGB lighting and smooth tracking.", 5, "2024-01-03", "RGB Mouse"),
            DummyDoc("Stopped working after a month.", 2, "2024-01-04", "Disappointing"),
            DummyDoc("Very lightweight and perfect for FPS games.", 4, "2024-01-05", "FPS Mouse"),
            DummyDoc("Bluetooth connection sometimes drops.", 3, "2024-01-06", "Bluetooth Issue"),
            DummyDoc("Customizable macros are a game changer.", 5, "2024-01-07", "Macro Mouse"),
            DummyDoc("DPI settings are easy to adjust.", 4, "2024-01-08", "DPI Mouse"),
            DummyDoc("Battery lasts for days, very happy.", 5, "2024-01-09", "Long Battery"),
            DummyDoc("A bit heavy for my taste, but works well.", 3, "2024-01-10", "Heavy Mouse"),
        ]

from tools import AgentTools

@pytest.fixture
def agent_tools():
    llm = DummyLLM()
    retriever = DummyRetriever()
    return AgentTools(llm, retriever)

def test_extract_important_keywords(agent_tools):
    # Updated to match new interface - only collection_name parameter
    keywords = agent_tools.extract_important_keywords("I want a wireless RGB mouse with good battery and macros")
    print("\n[TEST] extract_important_keywords prompt:\n", agent_tools.llm.last_prompt)
    print("[TEST] extract_important_keywords result:", keywords)
    assert isinstance(keywords, list)
    # Convert to lowercase for comparison since LLM might return different cases
    keywords_lower = [k.lower() for k in keywords]
    assert any("mouse" in k for k in keywords_lower)
    assert len(keywords) >= 5

def test_retrieve_useful_reviews(agent_tools):
    reviews = agent_tools.retrieve_useful_reviews(["mouse", "wireless"], k=10)
    print("\n[TEST] retrieve_useful_reviews result (first 2):", reviews[:2])
    assert isinstance(reviews, list)
    assert len(reviews) == 10
    assert "content" in reviews[0]
    assert "title" in reviews[0]
    assert "rating" in reviews[0]
    assert "date" in reviews[0]

def test_summarize_reviews(agent_tools):
    reviews = [
        {"title": "Awesome Mouse", "rating": 5, "date": "2024-01-01","content": "Great mouse, very responsive and ergonomic. I've been using it daily for both work and gaming, and the precision of the sensor is outstanding."},
        {"title": "Okay Mouse", "rating": 3, "date": "2024-01-02","content": "Not bad, but the battery life is quite short. On average, I have to recharge it every two or three days with moderate use."},
        {"title": "RGB Mouse", "rating": 5, "date": "2024-01-03","content": "Excellent RGB lighting and smooth tracking. The lighting effects are customizable through the companion software."},
        {"title": "Disappointing", "rating": 2, "date": "2024-01-04","content": "Stopped working after a month. At first, everything seemed fine — setup was easy, clicks were crisp, and tracking was precise."},
        {"title": "FPS Mouse", "rating": 4, "date": "2024-01-05","content": "Very lightweight and perfect for FPS games. The shape feels optimized for claw and fingertip grips."}
    ]
    summary = agent_tools.summarize_reviews(reviews)
    print("\n[TEST] summarize_reviews prompt:\n", agent_tools.llm.last_prompt[:500], "...")
    print("[TEST] summarize_reviews result:", summary)
    assert isinstance(summary, str)
    assert "Summary" in summary
    assert len(summary) > 50  # Ensure meaningful summary

def test_get_reviews_statistics(agent_tools):
    reviews = [
        {"content": "Great mouse, very responsive and ergonomic.", "rating": 5, "date": "2024-01-01", "title": "Awesome Mouse"},
        {"content": "Not bad, but battery life is short.", "rating": 3, "date": "2024-01-02", "title": "Okay Mouse"},
        {"content": "Excellent RGB lighting and smooth tracking.", "rating": 5, "date": "2024-01-03", "title": "RGB Mouse"},
        {"content": "Stopped working after a month.", "rating": 2, "date": "2024-01-04", "title": "Disappointing"},
        {"content": "Very lightweight and perfect for FPS games.", "rating": 4, "date": "2024-01-05", "title": "FPS Mouse"},
        {"content": "Bluetooth connection sometimes drops.", "rating": 3, "date": "2024-01-06", "title": "Bluetooth Issue"},
        {"content": "Customizable macros are a game changer.", "rating": 5, "date": "2024-01-07", "title": "Macro Mouse"},
        {"content": "DPI settings are easy to adjust.", "rating": 4, "date": "2024-01-08", "title": "DPI Mouse"},
        {"content": "Battery lasts for days, very happy.", "rating": 5, "date": "2024-01-09", "title": "Long Battery"},
        {"content": "A bit heavy for my taste, but works well.", "rating": 3, "date": "2024-01-10", "title": "Heavy Mouse"},
    ]
    stats = agent_tools.get_reviews_statistics(reviews)
    print("\n[TEST] get_reviews_statistics prompt:\n", agent_tools.llm.last_prompt[:500], "...")
    print("[TEST] get_reviews_statistics result:", stats)
    assert isinstance(stats, str)
    assert "Average rating" in stats or "average" in stats.lower()
    assert "10" in stats  # Should mention total count

def test_integration_workflow(agent_tools):
    """Test the complete workflow: extract -> retrieve -> summarize -> statistics"""
    print("\n[TEST] Testing complete workflow...")

    # Step 1: Extract keywords
    user_query = "I want a wireless gaming mouse with RGB lighting"
    keywords = agent_tools.extract_important_keywords(user_query)
    print(f"Step 1 - Keywords: {keywords}")
    assert isinstance(keywords, list)
    assert len(keywords) > 0

    # Step 2: Retrieve reviews
    reviews = agent_tools.retrieve_useful_reviews(keywords[:3], k=5)  # Use first 3 keywords
    print(f"Step 2 - Retrieved {len(reviews)} reviews")
    assert len(reviews) == 5

    # Step 3: Summarize
    summary = agent_tools.summarize_reviews(reviews)
    print(f"Step 3 - Summary length: {len(summary)} chars")
    assert len(summary) > 50

    # Step 4: Statistics
    stats = agent_tools.get_reviews_statistics(reviews)
    print(f"Step 4 - Stats: {stats[:100]}...")
    assert "Average" in stats or "average" in stats.lower()

    print("[TEST] Complete workflow successful!")
