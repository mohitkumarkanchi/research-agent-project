 **Research Agent Project** that integrates research paper ingestion, LlamaIndex semantic chunking with open-source embeddings, FAISS vector store, and Ollama Llama 3.1 local LLM with LangGraph for RAG.

# Research Agent Project

A cutting-edge AI/ML system to ingest research papers, semantically chunk their content with open-source embeddings, index the chunks in a FAISS vector store, and answer natural language queries using a local Llama 3.1 LLM server via LangGraph.

## Features

- **Research Paper Fetching:** Uses Semantic Scholar API for automated metadata and abstracts retrieval.
- **Semantic Chunking:** Uses LlamaIndex's `SemanticSplitterNodeParser` with open-source SentenceTransformers embeddings to generate meaningful document chunks.
- **Vector Search:** Employs FAISS for efficient approximate nearest neighbor search on chunk embeddings.
- **Local LLM Integration:** Integrates with a locally hosted Ollama Llama 3.1 model through LangGraph's `ChatOllama` client.
- **Agentic RAG Pipeline:** Combines retrieval and generation for high-quality, context-aware answers.
- **Configurable & Modular:** Easily customizable via YAML config files and well-structured modular Python code.
- **Production Ready:** Includes containerization support and Conda environment setup instructions.
  
---

## Directory Structure

```
research_agent_project/
│
├── agent/                        # Core modules (fetching, chunking, vector store, retrieval, agent)
│   ├── __init__.py
│   ├── agent.py                 # AI agent interacting with LLM and retriever
│   ├── chunker.py               # Semantic chunking with LlamaIndex + HuggingFace embeddings
│   ├── data_fetcher.py          # Research paper fetcher using Semantic Scholar API
│   ├── retriever.py             # Retrieves relevant chunks from vector store
│   └── vector_store.py          # FAISS vector index and embedding management
│
├── config/
│   └── config.yaml              # Runtime configurations
│
├── data/                        # (Optional) data storage folder
├── tests/                       # Tests (optional)
├── Dockerfile                   # Docker container setup
├── environment.yml              # Conda environment specification
├── main.py                     # Entry point script
└── README.md                   # Project overview and instructions (this file)
```

## Getting Started

### Prerequisites

- Install [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11+
- Docker (optional but recommended for containerized deployment)
- Ollama installed locally with Llama 3.1 model (instructions below)

### Setup Conda Environment

```bash
conda env create -f environment.yml
conda activate research_agent_env
```

### Start Ollama Llama 3.1 Local Server

Ensure Llama 3.1 is installed on Ollama and start the server:

```bash
ollama serve llama_3_1
```

Verify it’s accessible at the URL specified in [`config/config.yaml`](./config/config.yaml) (`http://localhost:11434` by default).

### Configuration

Edit `config/config.yaml` to adjust parameters:

```yaml
semantic_scholar_api: "https://api.semanticscholar.org/graph/v1/paper/search"
ollama_server_url: "http://localhost:11434"
embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"
chunk_size: 512
chunk_overlap: 50
faiss_index_path: "data/faiss.index"
```

### Running the Project

Start the interactive research query agent:

```bash
python main.py
```

Enter queries about research papers, e.g.:

```text
>> What are best software engineering practices for ML projects?
```

Type `exit` to quit.

## Detailed Workflow

1. **Fetch** research paper abstracts using Semantic Scholar API.
2. **Chunk** each abstract semantically using LlamaIndex’s `SemanticSplitterNodeParser` with open-source sentence transformer embeddings.
3. **Embed and index** chunks with FAISS vector store.
4. **Retrieve** top relevant chunks on user query from FAISS.
5. **Use** Ollama Llama 3.1 LLM to generate an answer based on retrieved context.

## Dependencies

- `requests`
- `sentence-transformers`
- `faiss-cpu`
- `llama-index`
- `langgraph`
- `langchain-ollama`
- `httpx`
- `pyyaml`
- `uvicorn`
- `structlog`
- `tenacity`

See full list in [environment.yml](./environment.yml).

## Docker Support

Build and run in Docker container:

```bash
docker build -t research-agent .
docker run -it --rm -p 8000:8000 research-agent
```

## Troubleshooting

- Make sure Ollama server is running and reachable.
- If FAISS or PyTorch raise errors, verify your environment setup.
- Semantic Scholar API enforces rate limits, adjust query limits accordingly.
- Verify Python environment matches installed packages (`conda activate research_agent_env`).

## Contribution & Extensions

- Add unit tests in the `tests/` folder.
- Integrate CI/CD pipelines using GitHub Actions.
- Enhance the frontend or add API endpoints.
- Support bigger datasets with batch processing and persistent storage.

## License

MIT License — Feel free to use, modify, and share!

## References

- [Semantic Scholar API](https://api.semanticscholar.org/)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [LangGraph](https://github.com/langgraph/langgraph)


