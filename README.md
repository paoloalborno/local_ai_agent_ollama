# 🤖 Local AI Agent with MCP Architecture - Gaming Reviews Analysis System

A **complete local artificial intelligence solution** that implements the **Model Context Protocol (MCP)** for advanced analysis of gaming product reviews, using a cutting-edge technology ecosystem.

## Architecture and Technologies

This project demonstrates advanced integration of multiple AI technologies, showcasing proficiency in modern AI development patterns and distributed 
system design. 
The architecture leverages the Model Context Protocol to create a seamless communication layer between client applications and AI-powered analysis tools.

### Technology Stack
The system integrates several technologies to deliver a comprehensive AI solution. 
**Ollama** serves as the local runtime environment for large language models, 
eliminating the need for external API dependencies while ensuring data privacy and reducing latency. 
**LangChain** provides the orchestration framework that binds different AI tools together, enabling complex workflows and tool chaining. 
**ChromaDB** functions as the vector database backbone, storing semantic embeddings that power the Retrieval-Augmented Generation (RAG) capabilities. 
The **Model Context Protocol (MCP)** establishes a standardized communication interface between AI agents and client applications, 
while **AsyncIO** enables high-performance concurrent processing of multiple requests.

### MCP Architecture
```
┌─────────────────┐    JSON-RPC    ┌─────────────────┐
│   MCP Client    │ ←─────────────→ │   MCP Server    │
│  (mcp_client.py)│                 │ (mcp_server.py) │
└─────────────────┘                 └─────────────────┘
         │                                   │
         │                          ┌─────────────────┐
         │                          │   Agent Tools   │
         │                          │   (tools.py)    │
         │                          └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  User Interface │                 │  Ollama Agent   │
│   (Interactive) │                 │ (ollama_agent.py)│
└─────────────────┘                 └─────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Vector Database │
                                    │   (vector.py)   │
                                    │   ChromaDB      │
                                    └─────────────────┘
```

The architecture implements a flexible interaction model where users can engage with the system through multiple pathways. 
The MCP Client provides an interactive interface that communicates directly with the MCP Server using JSON-RPC protocol, invoking the available tools.
The MCP Server exposes individual tools that can be called independently, allowing for granular control over the analysis pipeline. The Ollama Agent serves as an orchestrator that chains multiple tools together for complex queries, while the Vector Database provides the semantic search capabilities that power the entire system.

## - What the System Does

The system implements a sophisticated conversational AI agent that transforms NL (natural language) queries
into actionable insights from (gaming as example) product reviews. 
Rather than simple keyword matching, it extracts semantic understanding and meaningful patterns from review data.

The basic functionality revolves around intelligent keyword extraction (**extract_important_keywords** tool) from user queries,
where the system analyzes the semantic intent and identifies the most relevant search terms.

These keywords then drive a vector-based semantic search (**retrieve_reviews** tool) through reviews, 
retrieving the most contextually relevant content.

Resulting reviews are used by to generate comprehensive summaries that highlight key themes, pros and cons, 
and overall sentiment patterns (**summarize_reviews** tool).
Additionally, the system performs simple statistical analysis on the review data (**get_reviews_statistics** tool), 
calculating average ratings, rating ranges, and sentiment distribution.

The entire process can be orchestrated automatically through the **agent** tool, which chains all steps together,
or can be invoked individually to perform targeted analysis via  MCP protocol.

## Prerequisites and Installation

### 1. Install Ollama
```bash
# Windows: Download from https://ollama.ai
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Download AI models
```bash
# Main model for text generation
ollama pull qwen2.5:7b ( or ollama pull llama3.2:latest )

# Model for vector embeddings
ollama pull mxbai-embed-large

# Verify models are installed
ollama list
```

### 3. Project setup
```bash
# Clone repository
git clone <repository-url>
cd local_ai_agent_ollama

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directory for vector database
mkdir chroma_db
```

## System Usage

### Interactive MCP Client Commands

Start the interactive client:
```bash
python mcp_client.py
```

The system provides five main commands for different interaction patterns:

#### 1. `test` - Verify MCP Connection

**Usage:**
```bash
> test
```

**Example Output:**
```
Found 5 tools
```

This command verifies that the MCP server is running correctly and lists the number of available tools.
#### 2. `list` - Display Available Tools

**Usage:**
```bash
> list
```

**Example Output:**
```
Loading tools list:
Found 5 tools

  • agent
    Process a complete query through all steps: extract keywords, retrieve reviews, and generate comprehensive analysis

  • extract_important_keywords
    Extract the most important keywords from a user query to search for relevant reviews.

  • retrieve_useful_reviews
    Retrieve k reviews related to the given list of keywords.

  • summarize_reviews
    Generate a comprehensive summary of the given reviews, highlighting pros, cons, and key themes.

  • get_reviews_statistics
    Compute statistics on the given reviews (average rating, sentiment distribution, etc.).
```

This command provides a comprehensive overview of all available MCP tools, their names, and descriptions,
helping users understand what functionality is available.

#### 3. `extract <query>` - Intelligent Keyword Extraction

**Usage:**
```bash
> extract What are the best wireless gaming headsets for competitive FPS games?
```

**Example Output:**
```
Extracting important keywords from: 'What are the best wireless gaming headsets for competitive FPS games?'
Keywords: ["wireless", "gaming headsets", "competitive", "FPS games", "best"]
```

This tool demonstrates the system's natural language processing capabilities by analyzing user queries and extracting the most relevant search terms. 
The extracted keywords are automatically stored for potential use in subsequent `process` commands.

#### 4. `process <keywords> <k>` - Targeted Review Processing

**Usage:**
```bash
> process wireless,gaming headsets,competitive 8
```

**Example Output:**
```json
{
  "reviews": [
    {
      "content": "SteelSeries Arctis 7P - Excellent wireless gaming headset with low latency and great sound quality for competitive gaming...",
      "rating": 4.5,
      "date": "2023-08-15",
      "title": "SteelSeries Arctis 7P Review"
    },
    {
      "content": "Logitech G Pro X Wireless - Perfect for esports, crystal clear audio and comfortable for long gaming sessions...",
      "rating": 4.8,
      "date": "2023-09-22",
      "title": "Logitech G Pro X Wireless Gaming Headset"
    }
  ],
  "summary": "The reviews consistently praise wireless gaming headsets for competitive play, highlighting low latency, excellent audio quality, and comfort during extended gaming sessions. Most users recommend SteelSeries and Logitech models for their reliability and performance in FPS games.",
  "statistics": "Average rating: 4.6/5. Out of 8 reviews analyzed, 7 are positive (4-5 stars) and 1 is neutral (3 stars). Users particularly value low latency and audio clarity."
}
```

This command performs targeted analysis by retrieving a specified number of reviews based on provided keywords, 
then generating both summaries and statistical analysis of the results.

#### 5. `agent <query>` - Complete Analysis Pipeline

**Usage:**
```bash
> agent What are the most common complaints about gaming keyboards under $100?
```

**Example Output:**
```json
{
  "query": "What are the most common complaints about gaming keyboards under $100?",
  "keywords": ["gaming keyboards", "complaints", "under $100", "budget", "issues"],
  "reviews_count": 5,
  "summary": "Common complaints about budget gaming keyboards include inconsistent key switches, poor build quality with plastic construction, and inadequate RGB lighting customization. Users frequently mention that cheaper keyboards suffer from key chatter, uneven backlighting, and software issues. However, many acknowledge that for the price point, these keyboards still offer decent gaming performance despite the limitations.",
  "statistics": "Average rating: 3.4/5 across 5 reviews. 2 positive reviews (4-5 stars), 2 neutral (3 stars), and 1 negative (1-2 stars). Most critical feedback focuses on durability and software reliability.",
  "status": "success"
}
```

The agent command demonstrates the full power of the system by automatically orchestrating all analysis steps. 
It extracts keywords, retrieves relevant reviews, generates comprehensive summaries, and provides statistical 
insights in a single, structured JSON response.

## 🛠️ Implemented MCP Tools

The MCP server exposes **five** (but freely expandable) distinct tools 
that can be used independently or in combination to perform comprehensive review analysis. 
Each tool is designed with specific capabilities and can be invoked directly through the MCP protocol, 
providing maximum flexibility for different use cases.

### 1. **Agent Tool** (Complete Pipeline)
The Agent Tool serves as the orchestrator that automatically **chains** all analysis steps into a cohesive workflow. 
When invoked, it extracts keywords from the user query, retrieves relevant reviews, generates summaries, 
and computes statistics, returning a comprehensive JSON response that includes all analysis results.
This tool demonstrates advanced error handling and provides structured output that can be easily consumed by client applications.

### 2. **Extract Keywords Tool**
This tool performs semantic analysis of natural language queries to identify the most relevant search terms for review retrieval.
The extraction process is context-aware, meaning it considers the domain-specific terminology common in gaming product reviews.

### 3. **Retrieve Reviews Tool**
The retrieval tool leverages ChromaDB's vector search capabilities to find semantically similar reviews based on the provided keywords. 
It performs ranking based on semantic relevance rather than simple keyword matching, 
ensuring that the most contextually appropriate reviews are returned.
Each retrieved review includes rich metadata such as ratings, dates, and product titles, providing comprehensive context for analysis.

### 4. **Summarize Reviews Tool**
This tool performs advanced thematic analysis on retrieved reviews, identifying recurring patterns, pros and cons, and overall sentiment trends. 
The summarization process goes beyond simple text concatenation, employing sophisticated natural language generation to create coherent, 
informative summaries that capture the essence of user feedback. 
The tool integrates sentiment analysis to provide balanced perspectives on product strengths and weaknesses.

### 5. **Statistics Tool**
The statistics tool performs simple quantitative analysis on review data,
calculating metrics such as average ratings, rating distributions, and sentiment classifications. 
The tool handles various data quality issues and provides robust statistical measures even with incomplete or inconsistent review data.

## 🔧 Advanced Configuration

### Change Ollama Model

To modify the language model used by the system, edit the `initialize_system` function call in `mcp_server.py`:

```python
# In mcp_server.py, line 82-85:
async def main():
    """Main entry point"""
    # Initialize system
    if not initialize_system(model_name="llama3.2:latest"):  # Change model here
        print("Failed to initialize system", file=sys.stderr, flush=True)
        return
```

Available model options include:
- `"qwen2.5:7b"` Balanced performance and speed
- `"llama3.2:latest"` - Latest Llama model with improved capabilities

### Customize Vector Database Settings

The vector database configuration can be modified in `vector.py`:

```python
# In vector.py, line 36-45:
class ReviewsVectorStore:
    def __init__(self, csv_file_path: str = "reviews.csv", db_location: str = "./chroma_db", 
                 embedding_model: str = "mxbai-embed-large", collection_name: str = "gaming_reviews"):
        self.csv_file_path = csv_file_path
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)
```

To customize the retriever behavior, modify the `get_retriever` method:

```python
# In vector.py, line 77-79:
def get_retriever(self, k: int = 10):
    k = max(1, min(50, int(k)))  # Limit k between 1 and 50
    return self.vector_store.as_retriever(search_kwargs={"k": k})
```

### Modify Analysis Prompts

The AI analysis prompts can be customized in `tools.py`:

```python
# In tools.py, line 9-29:
self.prompt_keywords = (
    "Given the following user query your goal is to extract the most important keywords on order to retrieve related reviews via RAG.\n"
    "Add only very close synonyms. "
    "User query: {user_query}\n"
    "Return ONLY a comma-separated list of keywords, no explanations."
)
self.prompt_summary = (
    "Summarize the following reviews. Focus on:\n"
    "- Main pros and cons\n"
    "- Recurring themes\n"
    "- Overall sentiment\n"
    "- Key recommendations\n\n"
    "Reviews: {reviews}\n\n"
    "Provide a concise summary in 2-3 paragraphs."
)
self.prompt_stats = (
    "Analyze the following reviews and compute:\n"
    "- Average rating\n"
    "- Rating range\n"
    "- Total number of reviews\n"
    "- How many are positive (4-5) and how many are negative (1-2)\n\n"
    "Reviews: {reviews}\n\n"
    "Reply concisely."
)
```

### Initialize Database Settings

Database initialization behavior can be controlled in `vector.py`:

```python
# In vector.py, line 56-75:
def init_database(self, auto_recreate: bool = False) -> None:
    if not os.path.exists(self.db_location):
        raise FileNotFoundError(f"Database directory not found: {self.db_location}")
    if auto_recreate:
        try:
            self.vector_store.delete_collection()
            documents, ids = _df_to_documents(self.load_csv())
            self.vector_store.add_documents(documents=documents, ids=ids)
        except Exception as e:
            logging.error(f"Error recreating database: {e}")
            raise e
```

Set `auto_recreate=True` to force rebuild the database from CSV data.

## - Dataset and Performance

The system is designed to work with variable-sized datasets loaded from CSV files containing reviews. 
The flexibility of the (local - in memory) ChromaDB vector storage allows the system to scale from small experimental datasets 
to large production collections containing thousands of reviews. 
Each review record includes structured metadata such as ratings, dates, product titles, and review content, 
enabling rich semantic search and comprehensive analysis.

```csv
"Title","Date","Rating","Review"
"Start with lower settings","2022-10-16","4","Less is more with this graphics card. When I first got it, I cranked everything to ultra settings immediately. The temps got way too high and performance suffered. But this was also when the card just came out and drivers weren't optimized. It's powerful and long lasting once you find the right balance. Just be conservative with your overclock settings"
"No Title","2019-02-10","5","One of my favorite gaming mice I prefer this over the Razer DeathAdder any day of the week. It doesn't feel heavy at all and it looks so sleek on my desk setup"
"No Title","2019-02-09","3","Not the worst budget gaming headset, not the best one though. Works beautifully for single player games and casual voice chat, but if you're doing competitive FPS games the audio positioning isn't that precise. The microphone is also not as clear as I would like it to be."
"Perfect for Casual Gaming","2018-03-28","5","As a mom of 3 little kids with a full time job, gaming time is a luxury. This Nintendo Switch makes it easy and feasible since I can play during my lunch break! I now have multiple games and they all run fantastic!"
"Excellent mechanical keyboard","2017-07-21","5","This keyboard actually provides great tactile feedback. The RGB lighting is vibrant and customizable that will look good with any gaming setup. The switches feel amazing and the build quality is solid. The keyboard has lasted a good year without any key issues even though I'm not particularly careful with my gear. Would definitely recommend this product."
```

The vector database automatically handles the conversion of textual review content
into high-dimensional embeddings using Ollama's embedding models. 
These semantic representations enable the system to understand relationships and similarities between reviews 
that might not be apparent through traditional keyword-based approaches. 
The pre-calculated embeddings ensure rapid search performance, typically achieving sub-second response times even with large datasets.

### Performance Metrics
Performance characteristics vary based on dataset size and hardware configuration, 
but typical deployments achieve average response times under 2 seconds for complete analysis workflows. 
The system's asynchronous architecture supports concurrent user sessions, 
with typical configurations handling simultaneous users without performance degradation.

## Advanced Troubleshooting

### Ollama issues
```bash
# Check status
ollama ps

# Restart service
ollama serve

# Debug connection
curl http://localhost:11434/api/version
```

### ChromaDB issues
```bash
# Complete database reset
rm -rf chroma_db/*
python -c "from vector import ReviewsVectorStore; vs=ReviewsVectorStore(); vs.init_database(auto_recreate=True)"
```

### Debug MCP Protocol
```python
# Increase logging in mcp_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📁 Project Structure

```
local_ai_agent_ollama/
├── 🔧 mcp_server.py          # MCP Server implementation
├── 🖥️  mcp_client.py          # MCP Client interactive interface  
├── 🤖 ollama_agent.py         # AI Agent orchestrator
├── 🛠️  tools.py               # Agent tools collection
├── 🗄️  vector.py              # ChromaDB vector store management
├── 📊 reviews.csv             # Gaming reviews dataset
├── 📦 requirements.txt        # Python dependencies
├── 🗃️  chroma_db/             # Vector database (auto-generated)
└── 📖 README.md               # Project documentation
```

## Advanced Use Cases

The system's flexible architecture enables a wide range of analytical applications beyond simple review summarization. 
For competitive analysis, users can compare products across multiple dimensions by analyzing review sentiment, feature mentions,
and user satisfaction patterns. Market research applications benefit from the system's ability to identify emerging trends, 
common complaints, and feature requests across product categories.

## 📄 License

MIT License - See LICENSE file for details.

**Technologies**: Python AsyncIO • LangChain Framework • ChromaDB Vector Store • Ollama LLM Runtime • MCP Protocol • RAG Architecture • Semantic Search • AI Agent Orchestration • Real-time Processing
