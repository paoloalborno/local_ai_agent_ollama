import pytest

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
    def get_relevant_documents(self, query):
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

import json
from tools import AgentTools

@pytest.fixture
def agent_tools():
    llm = DummyLLM()
    retriever = DummyRetriever()
    return AgentTools(llm, retriever)

def test_extract_important_keywords(agent_tools):
    keywords = agent_tools.extract_important_keywords("I want a wireless RGB mouse with good battery and macros", "gaming_reviews")
    print("\n[TEST] extract_important_keywords prompt:\n", agent_tools.llm.last_prompt)
    print("[TEST] extract_important_keywords result:", keywords)
    assert isinstance(keywords, list)
    assert "mouse" in keywords
    assert len(keywords) >= 5

def test_retrieve_useful_reviews(agent_tools):
    reviews = agent_tools.retrieve_useful_reviews(["mouse", "wireless"], k=10)
    print("\n[TEST] retrieve_useful_reviews result (first 2):", reviews[:2])
    assert isinstance(reviews, list)
    assert len(reviews) == 10
    assert "content" in reviews[0]

def test_summarize_reviews(agent_tools):
    reviews = [
        {"title": "Awesome Mouse", "rating": 5, "date": "2024-01-01","content": "Great mouse, very responsive and ergonomic. I’ve been using it daily for both work and gaming, and the precision of the sensor is outstanding. The shape fits naturally in my hand, preventing wrist strain even after hours of use. The clicks feel tactile without being noisy, and the scroll wheel has just the right amount of resistance. Compared to my previous Logitech model, this one feels lighter and more accurate. Definitely worth the price — it’s one of those peripherals that instantly improves your workflow and comfort."},
        {"title": "Okay Mouse", "rating": 3, "date": "2024-01-02","content": "Not bad, but the battery life is quite short. On average, I have to recharge it every two or three days with moderate use, which is disappointing given the advertised endurance. The performance itself is decent — it tracks smoothly and connects quickly over Bluetooth — but the lack of endurance makes it less ideal for travel or long gaming sessions. It’s a fine product if you keep it plugged in often, but not something I’d recommend for heavy users."},
        {"title": "RGB Mouse", "rating": 5, "date": "2024-01-03","content": "Excellent RGB lighting and smooth tracking. The lighting effects are customizable through the companion software, allowing you to sync colors with your keyboard or even set per-zone brightness. Beyond aesthetics, the performance is top-notch — low latency, accurate movements, and a flawless sensor response. I use it mostly for gaming, and I’ve noticed improved aim consistency in FPS titles. The build quality feels premium, and the braided cable adds a touch of durability. Highly recommended for gamers who value both looks and performance."},
        {"title": "Disappointing", "rating": 2, "date": "2024-01-04","content": "Stopped working after a month. At first, everything seemed fine — setup was easy, clicks were crisp, and tracking was precise. Then, out of nowhere, the left click started to fail intermittently and eventually stopped registering completely. I tried updating firmware, resetting the connection, even testing it on another computer — nothing fixed it. For a product in this price range, such poor reliability is unacceptable. The support team responded slowly and offered only a partial replacement. Definitely a frustrating experience."},
        {"title": "FPS Mouse", "rating": 4, "date": "2024-01-05","content": "Very lightweight and perfect for FPS games. The shape feels optimized for claw and fingertip grips, and the low weight makes fast flicks effortless. I use it mostly for competitive shooters like Valorant and CS2, and it performs flawlessly. The DPI adjustment on the fly is a huge plus, letting me switch between precision aiming and fast movement. The only thing missing is a slightly better scroll wheel — it feels a bit loose. Still, it’s one of the best performance-to-price mice I’ve tested."},
        {"title": "Bluetooth Issue", "rating": 3, "date": "2024-01-06","content": "Bluetooth connection sometimes drops, especially when switching between multiple paired devices. It reconnects automatically, but the interruptions can be annoying during work. Apart from that, the mouse is very comfortable, and the performance over USB receiver mode is flawless. I appreciate the silent clicks and adjustable DPI range. If you plan to use it exclusively over Bluetooth, though, be prepared for occasional hiccups. With firmware updates, it might improve, but for now it’s just average."},
        {"title": "Macro Mouse", "rating": 5, "date": "2024-01-07","content": "Customizable macros are a game changer. I use the programmable buttons to automate repetitive tasks at work, like copy-paste or launching scripts, and it’s been a huge productivity boost. The companion software is intuitive, allowing me to set complex macros with delays and conditions. For gaming, it’s equally powerful — I mapped reloads and abilities to side buttons, and it feels seamless. The mouse is solidly built, with a premium matte finish and a comfortable grip. It’s one of those peripherals that blend professional and gaming use perfectly."},
        {"title": "DPI Mouse", "rating": 4, "date": "2024-01-08","content": "DPI settings are easy to adjust thanks to the dedicated button under the scroll wheel. The transition between sensitivity levels is smooth and instant, which is great when switching from general browsing to design work. The tracking feels consistent across surfaces — I’ve tried it on wood, cloth, and even glass with no issues. The build quality is decent, though the side grips could be slightly better. It’s a versatile mouse suitable for users who like fine control and quick customization."},
        {"title": "Long Battery", "rating": 5, "date": "2024-01-09","content": "Battery lasts for days, very happy with the performance. I’ve used it heavily for over a week without needing to recharge, and even with RGB lighting enabled, it’s incredibly power-efficient. The charging cable is USB-C and supports pass-through, so you can use it wired while charging — a thoughtful design. The weight balance feels great, and there’s zero input lag in wireless mode. Overall, it’s a reliable, high-end mouse that delivers on all promises, especially autonomy."},
        {"title": "Heavy Mouse", "rating": 3, "date": "2024-01-10","content": "A bit heavy for my taste, but it works well. The construction feels solid, almost too much — it’s made of metal and dense plastic, which adds durability but also weight. For long gaming sessions, especially FPS, it can feel tiring. However, for productivity tasks and editing, the extra stability actually helps with precision. The buttons are well placed, and the wheel is smooth. Overall, it’s a good choice for office use, less so for players who prefer ultra-light models."}
    ]
    summary = agent_tools.summarize_reviews(reviews)
    print("\n[TEST] summarize_reviews prompt:\n", agent_tools.llm.last_prompt[:500], "...")
    print("[TEST] summarize_reviews result:", summary)
    assert isinstance(summary, str)
    assert "Summary" in summary
    assert "battery" in summary.lower() or "rgb" in summary.lower()

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
    assert "Average rating" in stats
    assert "Total: 10" in stats or "Total number of reviews: 10" in stats
