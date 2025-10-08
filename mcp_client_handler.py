import json

class SimpleClientHandler:

    def __init__(self, client):
        self.client = client
        self.command_map = {
            "agent": self.handle_agent,
            "test": self.handle_test,
            "list": self.handle_list,
            "extract": self.handle_extract,
            "process": self.handle_process
        }

    async def handle_agent(self, parts):
        if len(parts) < 2:
            print("Usage: agent <query>")
            return
        user_query = parts[1]
        print(f"\nProcessing user_query: '{user_query}'")
        result = await self.client.call_tool("agent", {"user_query": user_query})
        if result:
            if "content" in result and result["content"]:
                json_data = json.loads(result["content"][0]["text"])
                print(json.dumps(json_data, indent=2, ensure_ascii=False))
            else:
                print(result)
        else:
            print("Agent not working")

    async def handle_test(self, parts):
        tools = await self.client.list_tools()
        if tools:
            print(f"Found {len(tools)} tools")
        else:
            print("Server not ready yet")

    async def handle_list(self, parts):
        print("\n Loading tools list:")
        tools = await self.client.list_tools()
        if tools:
            print(f"Found {len(tools)} tools")
            for tool in tools:
                print(f"\n  • {tool['name']}")
                print(f"    {tool['description']}")
        else:
            print("No tools available")

    async def handle_extract(self, parts):
        if len(parts) < 2:
            print("Usage: extract <query>")
            return
        user_query = parts[1]
        print(f"\nExtracting important keywords from: '{user_query}'")
        result = await self.client.call_tool("extract_important_keywords", {"user_query": user_query})
        if result and "content" in result:
            keywords = json.loads(result["content"][0]["text"])
            self.last_keywords = keywords
            print(f"Keywords: {keywords}")
        else:
            print("No keywords found")

    async def handle_process(self, parts):
        keywords = self.last_keywords
        k = 5
        if len(parts) > 1:
            keywords = parts[0].split(",")
            args = parts[1].split()
            k = int(args[1]) if len(args) > 1 else 5

        print(f"\nRetrieving {k} reviews given the following keywords: {keywords}")
        result = await self.client.call_tool("retrieve_useful_reviews", {"keywords": keywords, "k": k})
        if result and "content" in result:
            reviews = json.loads(result["content"][0]["text"])
            summary = await self.client.call_tool("summarize_reviews", {"reviews": reviews})
            statistics = await self.client.call_tool("get_reviews_statistics", {"reviews": reviews})
            print(f"\nFound {len(reviews)} reviews:")
            json_result = {"reviews": reviews, "summary": summary, "statistics": statistics}
            print(json.dumps(json_result, indent=2))
        else:
            print("No reviews found")