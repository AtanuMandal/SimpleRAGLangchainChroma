# RAG-01 — Simple RAG with LangChain + Chroma + OpenAI

This project implements a **Retrieval-Augmented Generation (RAG)** system using modern **LCEL** (LangChain Expression Language) patterns. It ingests PDF and text files into a Chroma vector store and provides an interactive chat interface powered by OpenAI's GPT models.

## Project Structure

```
rag_01/
├── main.py              # Main RAG application with ingestion and chat
├── pyproject.toml       # Project dependencies and configuration
├── README.md            # This file
└── data/
    └── wiki-deep-learning.txt  # Sample document for ingestion
```

## Features

- **Document Loading**: Automatically loads PDFs and text files from a directory
- **Text Chunking**: Uses `RecursiveCharacterTextSplitter` for intelligent document splitting
- **Embeddings**: HuggingFace embeddings (sentence-transformers/all-mpnet-base-v2)
- **Vector Store**: Chroma for persistent vector storage and retrieval
- **LCEL-based Chat**: Modern LangChain Expression Language pipeline with GPT-4o-mini
- **Context-aware QA**: Retrieves top-4 relevant documents and answers based on provided context

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -U pip
pip install -e .
```

This installs packages defined in `pyproject.toml`, including:
- langchain, langchain-community, langchain-core
- langchain-huggingface, langchain-openai
- python-dotenv
- chromadb

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

## Usage

### Ingest Documents

Place your PDF and text files in the `data/` directory, then run:

```bash
python main.py --ingest --data_dir ./data --persist_dir ./chroma_db
```

**Optional parameters:**
- `--chunk_size` (default: 1000): Size of each text chunk
- `--chunk_overlap` (default: 200): Overlap between chunks

Example:
```bash
python main.py --ingest --data_dir ./data --persist_dir ./chroma_db --chunk_size 1500 --chunk_overlap 300
```

### Start Chat

Once documents are ingested:

```bash
python main.py --persist_dir ./chroma_db
```

Type your questions and the RAG system will retrieve relevant context and answer based on your documents.

Type `exit` or `quit` to end the conversation.

**Optional parameters:**
- `--temperature` (default: 0.0): Controls response creativity (0.0 = deterministic, higher = more creative)

Example:
```bash
python main.py --persist_dir ./chroma_db --temperature 0.5
```

## Key Functions

- **`load_documents_from_dir(data_dir)`**: Loads all PDFs and text files from directory
- **`chunk_documents(docs, chunk_size, chunk_overlap)`**: Splits documents into manageable chunks
- **`ingest(...)`**: Orchestrates loading, chunking, and storing in Chroma
- **`start_chat(...)`**: Runs interactive LCEL-based RAG conversation
- **`format_docs(docs)`**: Formats retrieved documents for the prompt

## Technologies

- **LangChain**: LLM orchestration and LCEL pipelines
- **Chroma**: Vector database for semantic search
- **HuggingFace**: Open-source embeddings model
- **OpenAI API**: GPT-4o-mini for question answering
- **Python 3.10+**

## Notes

- The system retrieves the top 4 most relevant documents for each query
- Responses are strictly grounded in provided documents
- First ingestion will create the Chroma DB; subsequent runs update it
- Ensure `OPENAI_API_KEY` is set before running chat
