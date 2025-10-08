import asyncio
import json
import sys
import traceback
from typing import Dict, Any, List
import mcp.types as types
import mcp.server.stdio
from langchain_ollama import OllamaLLM
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from agent import Agent
from tools import AgentTools
from vector import ReviewsVectorStore

server = Server("reviews-agent")
llm = None
vector_store: ReviewsVectorStore = None
retriever = None
tools: AgentTools = None
agent: Agent = None

tool_handlers = {
    "extract_important_keywords": lambda args: tools.extract_important_keywords(args["user_query"]),
    "retrieve_useful_reviews": lambda args: tools.retrieve_useful_reviews(args["keywords"], args.get("k", 5)),
    "summarize_reviews": lambda args: tools.summarize_reviews(args["reviews"]),
    "get_reviews_statistics": lambda args: tools.get_reviews_statistics(args["reviews"])
}

def initialize_system(model_name: str = "llama3.2:latest", k: int = 5) -> bool:
    """Initialize all components needed for the MCP server
        - Ollama LLM, a ReviewVectorStore, a RAG retriever, AgentTools instance and an Agent instance
    """
    
    global llm, vector_store, retriever, tools, agent # they are global because they are used in the @server.list_tools and @server.call_tool decorators

    try:
        print(f"Loading model: {model_name}", file=sys.stderr)
        llm = OllamaLLM(model=model_name)

        print("Initializing vector database...", file=sys.stderr)
        vector_store = ReviewsVectorStore()
        vector_store.init_database(auto_recreate=False)

        print("Create a RAG retriever...", file=sys.stderr)
        retriever = vector_store.get_retriever(k=k)

        print("Initializing tools...", file=sys.stderr)
        tools = AgentTools(llm, retriever)

        print("Initializing agent...", file=sys.stderr)
        agent = Agent(llm, retriever)

        print("System initialized successfully", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Error initializing system: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

# the following decorated methods are part of the MCP framework - the name of the method is dynamic (list tools and call tool are chosen by the dev)

@server.list_tools() # tool provider (communicate to server the list of available tools)
async def handle_list_tools() -> List[types.Tool]:  # here async is needed because the decorator expects an async function
    if not tools:
        return []
    return [
        types.Tool(
            name="agent",
            description="Process a complete query through all steps: extract keywords, retrieve reviews, and generate summary and statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user's question or query to process"
                    },
                    "k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of reviews to retrieve"
                    }
                },
                "required": ["user_query"]
            }
        ),
        types.Tool(
            name="extract_important_keywords",
            description="Extract the most important keywords from a user query. Keywords are then used to search for related reviews.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user's question or query"
                    }
                },
                "required": ["user_query"]
            }
        ),
        types.Tool(
            name="retrieve_useful_reviews",
            description="Retrieve k reviews related to the given list of keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords to search for"
                    },
                    "k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of reviews to retrieve"
                    }
                },
                "required": ["keywords"]
            }
        ),
        types.Tool(
            name="summarize_reviews",
            description="Generate a comprehensive summary of the given reviews, highlighting pros, cons, and key themes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "reviews": {
                        "type": "array",
                        "description": "List of reviews to summarize"
                    }
                },
                "required": ["reviews"]
            }
        ),
        types.Tool(
            name="get_reviews_statistics",
            description="Compute statistics on the given reviews (average rating, sentiment distribution, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "reviews": {
                        "type": "array",
                        "description": "List of reviews to analyze"
                    }
                },
                "required": ["reviews"]
            }
        )
    ]

@server.call_tool() # executor of a tool - When server receives a request to call a tool,
                    # this method receives the tool name and arguments, executes the tool, and returns the result
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    try:
        if not tools:
            return [types.TextContent(type="text", text=json.dumps({"error": "Agent tools not initialized"}))]
        if name in tool_handlers:
            result = tool_handlers[name](arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        else:
            return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    if not initialize_system():
        print("Failed to initialize the server components", file=sys.stderr, flush=True)
        return

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="reviews-agent",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())