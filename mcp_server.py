import asyncio
import json
import sys
from typing import Dict, Any, List
import mcp.types as types
import mcp.server.stdio
from langchain_ollama import OllamaLLM
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from ollama_agent import OllamaAgent
from tools import AgentTools
from vector import ReviewsVectorStore

server = Server("reviews-agent")
llm = None
vector_store: ReviewsVectorStore = None
retriever = None
agent_tools: AgentTools = None
ollama_agent: OllamaAgent = None

def initialize_system(model_name: str = "llama3.2:latest", k: int = 5) -> bool:
    """Initialize all components needed for the MCP server"""
    global llm, vector_store, retriever, agent_tools, ollama_agent

    try:
        # Load LLM
        print(f"Loading model: {model_name}", file=sys.stderr)
        llm = OllamaLLM(model=model_name)

        # Initialize vector store
        print("Initializing vector store...", file=sys.stderr)
        vector_store = ReviewsVectorStore()
        vector_store.init_database(auto_recreate=False)

        # Create retriever
        retriever = vector_store.get_retriever(k=k)

        # Initialize agent tools
        agent_tools = AgentTools(llm, retriever)
        ollama_agent = OllamaAgent(llm, retriever)

        print("System initialized successfully", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Error initializing system: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="agent",
            description="Process a complete query through all steps: extract keywords, retrieve reviews, and generate comprehensive analysis",
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
            description="Extract the most important keywords from a user query to search for relevant reviews.",
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

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    try:
        if not agent_tools:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Agent tools not initialized"})
            )]

        # Execute the requested tool
        if name == "extract_important_keywords":
            user_query = arguments["user_query"]
            result = agent_tools.extract_important_keywords(user_query)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False)
            )]

        elif name == "retrieve_useful_reviews":
            global retriever
            keywords_list = arguments["keywords"]
            k = arguments.get("k", 5)

            if retriever:
                retriever = vector_store.get_retriever(k=k)
                agent_tools.retriever = retriever

            result = agent_tools.retrieve_useful_reviews(keywords_list, k)
            return [types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False)
            )]

        elif name == "summarize_reviews":
            reviews = arguments["reviews"]
            result = agent_tools.summarize_reviews(reviews)
            return [types.TextContent(
                type="text",
                text=str(result)
            )]

        elif name == "get_reviews_statistics":
            reviews = arguments["reviews"]
            result = agent_tools.get_reviews_statistics(reviews)
            return [types.TextContent(
                type="text",
                text=str(result)
            )]

        elif name == "agent":
            if not ollama_agent:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "Ollama agent not initialized"})
                )]
            print(f"Processing query: {arguments['user_query']}", file=sys.stderr)
            result = ollama_agent.process_query(arguments["user_query"])
            print(f"Query completed", file=sys.stderr, flush=True)
            return [types.TextContent(
                type="text",
                text=result
            )]

        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]

    except Exception as e:
        import traceback
        error_msg = f"Error executing {name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def main():
    """Main entry point"""
    # Initialize system
    if not initialize_system():
        print("Failed to initialize system", file=sys.stderr, flush=True)
        return

    # Start MCP server
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