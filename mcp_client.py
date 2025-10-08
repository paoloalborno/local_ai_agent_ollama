import asyncio
import json
import sys
import traceback
from mcp_client_handler import SimpleClientHandler

class SimpleMCPClient:

    def __init__(self):
        self.process = None # this internal field is used to store the subprocess, this is needed to start/stop the server
        self.request_id = 0 # request id is needed to identify the response to a request
        self.last_keywords = []

    async def start_server(self):
        print("Server is starting...")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, 'mcp_server.py',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        print("Waiting for server initialization...")
            # this part is needed to waits for any initial output from the server, which may include error messages or status updates
        try:
            stderr_output = await asyncio.wait_for(
                self.process.stderr.read(1024),
                timeout=10.0
            )
            stderr_text = stderr_output.decode().strip()
            if stderr_text:
                print(f"Output from MCP server: {stderr_text}")
        except asyncio.TimeoutError:
            print("No messages from MCP server")
        await asyncio.sleep(1)

        print("Trying to connect to MCP server...")
        """ 
            Start of JSON-RPC communication - Remote Procedure Calls (RPCs)
            This part sends an "initialize" request to the server, which is a standard part of the JSON-RPC protocol for language servers
            it includes information about the client, such as its name and version
            the server is expected to respond with its capabilities and other initialization details
            if the initialization is successful, a notification is sent to the server indicating that the client is ready
            if the initialization fails, an error message is printed and the function returns False 
        """

        init_response = await self._send_request(
            "initialize",
            "initialize-server",{
            "protocolVersion": "202s4-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "simple-client",
                "version": "1.0.0"
            }
        })

        if init_response:
            print("Server running...")
            await self._send_notification("notifications/initialized") # notifies the server that the client is ready to start communicating. It does not wait for a response.
            return True
        else:
            print("Server not responding - exiting...")
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
        request = { "jsonrpc": "2.0", "id": self.request_id, "method": method}
        if params:
            request["params"] = params

        request_str = json.dumps(request) + "\n"

        print(f"Sending: {method + " " + method_name}")
        self.process.stdin.write(request_str.encode()) # here is the actual request sent to the server (it is done via stdin - standard input of the process running in self.process
        await self.process.stdin.drain() # ensures that the request is sent to the server, it cleans the write buffer

        try:
            response_line = await asyncio.wait_for( self.process.stdout.readline(), timeout=timeout)
            if response_line:
                response_text = response_line.decode().strip()
                try:
                    response = json.loads(response_text)
                    return response
                except Exception as e:
                    print(f"Error parsing JSON response: {e}")
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
        response = await self._send_request("tools/call", name,{ "name": name, "arguments": arguments }, timeout)
        if response and "result" in response:
            return response["result"]
        return None

    async def stop_server(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()



async def interactive_client():
    client = SimpleMCPClient()
    handler = SimpleClientHandler(client)
    try:
        success = await client.start_server()
        if not success:
            return

        print("\n" + "=" * 60)
        print("Commands:")
        print("=" * 60)
        print("1. test                          - Test MPC connection")
        print("2. list                          - List MCP server tools")
        print("3. agent - <query>               - Processes user query using the agent (if available)")
        print("4. extract - <query>             - Extract keywords")
        print("5. process - <keywords> - <k>    - Process k reviews (retrieve, summarize and compute statistics) given a list of comma-separated keywords")
        print("6. quit                          - Quit")
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

            parts = [p.strip() for p in user_input.split("-")]
            command = parts[0]

            try:
                if command not in handler.command_map:
                    print("Command not recognized.")
                    continue
                else:
                    await handler.command_map[command](parts)
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n Bye ")
    finally:
        print("\n Shutting down server...")
        await client.stop_server()


if __name__ == "__main__":
    asyncio.run(interactive_client())