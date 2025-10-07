import asyncio
import json
import sys

class SimpleMCPClient:

    def __init__(self):
        self.process = None
        self.request_id = 0
        self.last_keywords = []
        self.last_processed_reviews = {}

    async def start_server(self):
        print("Server is starting...")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, 'mcp_server.py',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        print("Waiting for server initialization...")
        try:
            stderr_output = await asyncio.wait_for(
                self.process.stderr.read(1024),
                timeout=5.0
            )
            stderr_text = stderr_output.decode().strip()
            if stderr_text:
                print(f"Output from MCP server: {stderr_text}")
        except asyncio.TimeoutError:
            print("No messages from MCP server")
        await asyncio.sleep(1)

        print("Trying to connect to MCP server...")
        init_response = await self._send_request("initialize", "initialize",{
            "protocolVersion": "202s4-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "simple-client",
                "version": "1.0.0"
            }
        })

        if init_response:
            print("init done")
            await self._send_notification("notifications/initialized")
            return True
        else:
            print("init failed")
            return False

    async def _send_notification(self, method, params=None):
        notification = { "jsonrpc": "2.0", "method": method }
        if params:
            notification["params"] = params

        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()

    async def _send_request(self, method, method_name, params=None, timeout=10.0):
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        request_str = json.dumps(request) + "\n"

        print(f"Sending: {method + " " + method_name}")
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        try:
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
            if response_line:
                response_text = response_line.decode().strip()
                if response_text:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON response")
                        return None
            else:
                print("Empty response from MCP server")
        except asyncio.TimeoutError:
            print("Waiting for response timed out")
            if self.process.returncode is not None:
                print(f"Server shut down - code: {self.process.returncode}")

        return None

    async def list_tools(self):
        response = await self._send_request("tools/list", "list")
        if response and "result" in response:
            return response["result"].get("tools", [])
        return []

    async def call_tool(self, name, arguments):
        timeout = 60.0 if name == "agent" else 10.0

        response = await self._send_request("tools/call", name,{
            "name": name,
            "arguments": arguments
        }, timeout)
        if response and "result" in response:
            return response["result"]
        return None

    async def stop_server(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()


async def interactive_client():
    client = SimpleMCPClient()
    try:
        print("Starting MCP...\n")
        success = await client.start_server()

        if not success:
            print("\n Failed to start MCP server.")
            return

        print("\n" + "=" * 60)
        print("Commands:")
        print("=" * 60)
        print("1. test                      - Test MPC connection")
        print("2. list                      - List MCP server tools")
        print("3. agent <query>             - Processes user query using the agent (if available)")
        print("4. extract <query>           - Extract keywords")
        print("5. process <keywords> <k>    - Process k reviews (retrieve, summarize and compute statistics) given a list of comma-separated keywords")
        print("6. quit                      - Quit")
        print("=" * 60)
        print("=" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "quit":
                break

            parts = user_input.split(" ", 1)
            command = parts[0]

            try:
                if command == "agent":
                    user_query = parts[1]
                    print(f"\nProcessing user_query: '{user_query}'")
                    result = await client.call_tool(
                        "agent",
                        {"user_query": user_query}
                    )
                    if result:
                        if "content" in result and result["content"]:
                            json_data = json.loads(result["content"][0]["text"])
                            print(json.dumps(json_data, indent=2, ensure_ascii=False))
                        else:
                            print(result)
                    else:
                        print("Agent not working")

                elif command == "test":
                    tools = await client.list_tools()
                    if tools:
                        print(f"Found {len(tools)} tools")
                    else:
                        print("Server not ready yet")

                elif command == "list":
                    print("\n Loading tools list:")
                    tools = await client.list_tools()
                    if tools:
                        print(f"Found {len(tools)} tools")
                        for tool in tools:
                            print(f"\n  • {tool['name']}")
                            print(f"    {tool['description']}")
                    else:
                        print("No tools available")

                elif command == "extract" and len(parts) > 1:
                    user_query = parts[1]
                    print(f"\nExtracting important keywords from: '{user_query}'")
                    result = await client.call_tool(
                        "extract_important_keywords",
                        {"user_query": user_query}
                    )
                    if result and "content" in result:
                        keywords = json.loads(result["content"][0]["text"])
                        client.last_keywords = keywords
                        print(f"Keywords: {keywords}")
                    else:
                        print("No keywords found")

                elif command == "process":
                    keywords = client.last_keywords
                    k = 5
                    if len(parts) > 1:
                        keywords = parts[0].split(",")
                        args = parts[1].split()
                        k = int(args[1]) if len(args) > 1 else 5

                    print(f"\nRetrieving {k} reviews given the following keywords: {keywords}")
                    result = await client.call_tool(
                        "retrieve_useful_reviews",
                        {"keywords": keywords, "k": k}
                    )
                    if result and "content" in result:
                        reviews = json.loads(result["content"][0]["text"])
                        summary = await client.call_tool(
                            "summarize_reviews",
                            {"reviews": reviews}
                        )
                        statistics = await client.call_tool(
                            "get_reviews_statistics",
                            {"reviews": reviews}
                        )
                        print(f"\nFound {len(reviews)} reviews:")
                        json_result = {"reviews": reviews, "summary": summary, "statistics": statistics}
                        print(json.dumps(json_result, indent=2))

                    else:
                        print("No reviews found")

                else:
                    print("Command not recognized.")

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n Bye ")
    finally:
        print("\n Shutting down server...")
        await client.stop_server()


if __name__ == "__main__":
    asyncio.run(interactive_client())