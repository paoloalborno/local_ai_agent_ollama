# Local AI Agent with Ollama - Product Reviews Q&A System

## Project Description

This project implements an intelligent question-answering system to analyze e-commerce product reviews using local AI models via Ollama. The system allows you to ask natural language questions about reviews and get accurate answers based on the data.

## Technologies Used

- **Ollama**: Local runtime for LLM (Large Language Models)
- **LangChain**: Framework for developing LLM applications
- **ChromaDB**: Vector database for semantic search
- **Pandas**: CSV data processing and analysis
- **Python 3.12+**: Main programming language

### Supported AI Models
- **Qwen3:8b**: Optimized multilingual model
- **Llama3.2:latest**: Meta AI model for conversations

## Project Purpose

The system is designed to:
- Analyze large volumes of product reviews
- Provide intelligent answers to specific product questions
- Use semantic search to find relevant reviews
- Operate completely offline with local AI models

## Installation and Setup

### Prerequisites
1. **Install Ollama**: Download from [ollama.com](https://ollama.com)
2. **Download required models**:
   ```bash
   ollama pull qwen3:8b
   ollama pull llama3.2:latest
   ollama pull mxbai-embed-large
   ```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Place the `reviews.csv` file in the project directory
2. The file must contain columns: `Title`, `Date`, `Rating`, `Review`

## Usage

1. **Start the system**:
   ```bash
   python app.py
   ```

2. **Select the model** when prompted (1 for Qwen3, 2 for Llama3.2)

3. **Configure the vector database** when prompted

4. **Ask questions** about products in natural language

### Example Questions
- "What are the products with the best reviews?"
- "Are there common problems mentioned in the reviews?"
- "Which products last the longest according to reviews?"
- "What do customers think about the value for money?"

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   reviews.csv   │────│   vector.py      │────│   ChromaDB      │
│   (Input Data)  │    │   (Processing)   │    │   (Storage)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     app.py      │────│    LangChain     │────│     Ollama      │
│   (Interface)   │    │    (Pipeline)    │    │   (Local LLM)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## File Structure

- `app.py`: User interface and main Q&A logic
- `vector.py`: Vector database and embeddings management
- `reviews.csv`: Product reviews dataset
- `requirements.txt`: Python project dependencies
- `chrome_langchain_db/`: ChromaDB vector database directory
