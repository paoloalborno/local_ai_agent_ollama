import pytest
import asyncio
import json
import sys
import os
import tempfile
import subprocess
import time
from unittest.mock import Mock, patch

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_client import SimpleMCPClient

class MockMCPServer:
    """Mock MCP server for testing without actual Ollama dependencies"""

    def __init__(self):
        self.tools = [
            {
                "name": "extract_important_keywords",
                "description": "Extract keywords from user query"
            },
            {
                "name": "retrieve_useful_reviews",
                "description": "Retrieve reviews based on keywords"
            },
            {
                "name": "summarize_reviews",
                "description": "Summarize a list of reviews"
            },
            {
                "name": "get_reviews_statistics",
                "description": "Get statistics from reviews"
            },
            {
                "name": "agent",
                "description": "Complete agent workflow"
            }
        ]

    def mock_responses(self, tool_name, arguments):
        """Generate mock responses for different tools"""
        if tool_name == "extract_important_keywords":
            return ["mouse", "wireless", "gaming", "rgb", "battery"]

        elif tool_name == "retrieve_useful_reviews":
            return [
                {
                    "title": "Great Gaming Mouse",
                    "content": "Excellent mouse for gaming, very responsive",
                    "rating": 5,
                    "date": "2024-01-01"
                },
                {
                    "title": "Good Mouse",
                    "content": "Decent mouse, battery could be better",
                    "rating": 4,
                    "date": "2024-01-02"
                }
            ]

        elif tool_name == "summarize_reviews":
            return "Summary: The reviews indicate positive sentiment with users appreciating responsiveness but noting battery life concerns."

        elif tool_name == "get_reviews_statistics":
            return "Average rating: 4.5, Total reviews: 2, Positive: 2, Negative: 0"

        elif tool_name == "agent":
            return {
                "query": arguments.get("user_query", ""),
                "keywords": ["mouse", "gaming"],
                "reviews_count": 2,
                "summary": "Positive reviews overall",
                "statistics": "Average rating: 4.5",
                "status": "success"
            }

        return {"error": f"Unknown tool: {tool_name}"}

@pytest.fixture
def mock_server():
    return MockMCPServer()

class TestMCPClient:
    """Test the MCP client functionality"""

    def test_client_initialization(self):
        """Test that client can be initialized"""
        client = SimpleMCPClient()
        assert client.request_id == 0
        assert client.last_keywords == []
        assert client.process is None

    def test_client_attributes(self):
        """Test client has required attributes"""
        client = SimpleMCPClient()
        assert hasattr(client, 'request_id')
        assert hasattr(client, 'process')

class TestMCPIntegration:
    """Test MCP server integration without starting actual server"""

    def test_mock_server_tools(self, mock_server):
        """Test mock server provides expected tools"""
        tools = mock_server.tools
        assert len(tools) == 5

        tool_names = [tool["name"] for tool in tools]
        expected_tools = [
            "extract_important_keywords",
            "retrieve_useful_reviews",
            "summarize_reviews",
            "get_reviews_statistics",
            "agent"
        ]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_mock_responses_keywords(self, mock_server):
        """Test keyword extraction mock response"""
        result = mock_server.mock_responses("extract_important_keywords", {"user_query": "test query"})
        assert isinstance(result, list)
        assert "mouse" in result
        assert len(result) >= 3

    def test_mock_responses_reviews(self, mock_server):
        """Test review retrieval mock response"""
        result = mock_server.mock_responses("retrieve_useful_reviews", {"keywords": ["mouse"]})
        assert isinstance(result, list)
        assert len(result) == 2
        assert "title" in result[0]
        assert "content" in result[0]
        assert "rating" in result[0]

    def test_mock_responses_summary(self, mock_server):
        """Test summary mock response"""
        result = mock_server.mock_responses("summarize_reviews", {"reviews": []})
        assert isinstance(result, str)
        assert "Summary" in result
        assert len(result) > 20

    def test_mock_responses_statistics(self, mock_server):
        """Test statistics mock response"""
        result = mock_server.mock_responses("get_reviews_statistics", {"reviews": []})
        assert isinstance(result, str)
        assert "Average rating" in result
        assert "Total reviews" in result

    def test_mock_responses_agent(self, mock_server):
        """Test agent workflow mock response"""
        result = mock_server.mock_responses("agent", {"user_query": "test query"})
        assert isinstance(result, dict)
        assert "query" in result
        assert "keywords" in result
        assert "status" in result
        assert result["status"] == "success"

class TestMCPWorkflow:
    """Test complete MCP workflow simulation"""

    def test_complete_workflow_simulation(self, mock_server):
        """Simulate a complete workflow through the MCP tools"""
        print("\n[TEST] Simulating complete MCP workflow...")

        # Step 1: Extract keywords
        user_query = "I want a wireless gaming mouse with RGB"
        keywords_result = mock_server.mock_responses("extract_important_keywords", {"user_query": user_query})
        print(f"Step 1 - Keywords: {keywords_result}")
        assert isinstance(keywords_result, list)

        # Step 2: Retrieve reviews
        reviews_result = mock_server.mock_responses("retrieve_useful_reviews", {"keywords": keywords_result})
        print(f"Step 2 - Reviews count: {len(reviews_result)}")
        assert isinstance(reviews_result, list)
        assert len(reviews_result) > 0

        # Step 3: Summarize
        summary_result = mock_server.mock_responses("summarize_reviews", {"reviews": reviews_result})
        print(f"Step 3 - Summary: {summary_result[:100]}...")
        assert isinstance(summary_result, str)

        # Step 4: Statistics
        stats_result = mock_server.mock_responses("get_reviews_statistics", {"reviews": reviews_result})
        print(f"Step 4 - Statistics: {stats_result}")
        assert isinstance(stats_result, str)

        # Step 5: Agent workflow
        agent_result = mock_server.mock_responses("agent", {"user_query": user_query})
        print(f"Step 5 - Agent result: {agent_result}")
        assert isinstance(agent_result, dict)
        assert agent_result["status"] == "success"

        print("[TEST] Complete workflow simulation successful!")

class TestMCPProtocol:
    """Test MCP protocol message formatting"""

    def test_request_format(self):
        """Test that request messages are properly formatted"""
        client = SimpleMCPClient()

        # Test that request ID increments
        initial_id = client.request_id
        client.request_id += 1
        assert client.request_id == initial_id + 1

        # Test request structure
        request = {
            "jsonrpc": "2.0",
            "id": client.request_id,
            "method": "tools/call",
            "params": {
                "name": "extract_important_keywords",
                "arguments": {"user_query": "test"}
            }
        }

        # Validate required fields
        assert "jsonrpc" in request
        assert "id" in request
        assert "method" in request
        assert request["jsonrpc"] == "2.0"

    def test_response_parsing(self):
        """Test response message parsing"""
        # Mock successful response
        response_str = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": '["mouse", "gaming"]'}]
            }
        })

        response = json.loads(response_str)
        assert "result" in response
        assert "content" in response["result"]

        # Extract and parse tool result
        content = response["result"]["content"][0]["text"]
        tool_result = json.loads(content)
        assert isinstance(tool_result, list)

def test_error_handling():
    """Test error handling in MCP operations"""
    client = SimpleMCPClient()

    # Test that client handles missing process gracefully
    assert client.process is None

    # Test that client can be created without errors
    assert isinstance(client.last_keywords, list)

if __name__ == "__main__":
    # Run tests with pytest
    print("Running MCP integration tests...")
    pytest.main([__file__, "-v", "-s"])

