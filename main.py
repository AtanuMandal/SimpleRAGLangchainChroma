import os
import argparse
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def load_documents_from_dir(data_dir: str):
    loaders  = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.lower().endswith(".pdf"):
                loaders.append(PyPDFLoader(path))
            elif fname.lower().endswith(('.txt', '.md')):
                loaders.append(TextLoader(path, encoding='utf8'))
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
           
        except Exception as e:
            print(f"Failed to load documents from {loader.file_path}, skipping.")
            print(f"An error occurred: {e}")
            continue
    return docs


def chunk_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def ingest(data_dir: str, persist_directory: str, chunk_size: int, chunk_overlap: int):
    print(f"Loading documents from: {data_dir}")
    docs = load_documents_from_dir(data_dir)
    if not docs:
        print("No documents found to ingest.")
        return
    print(f"Found {len(docs)} raw documents. Splitting into chunks...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(chunks)} document chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Creating/Updating Chroma vector store...")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory, collection_name="docs")
    vectordb.persist()
    print(f"Ingestion complete. Chroma persisted at: {persist_directory}")


def start_chat(persist_directory: str, temperature: float = 0.0):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if not os.path.isdir(persist_directory):
        print(f"Chroma DB not found at {persist_directory}. Run with --ingest first.")
        return
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="docs")
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    
    # LCEL-based RAG chain
    template = """You are a strict RAG assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."

 context:
{context}

Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    print("Chat ready. Type your question (or 'exit' to quit).")
    while True:
        try:
            query = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
        if not query:
            continue
        if query.strip().lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        try:
            result = chain.invoke(query)
            answer = result.content if hasattr(result, 'content') else str(result)
            print("Assistant:", answer)
        except Exception as e:
            print(f"Error processing query: {e}")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    load_dotenv()
    print(os.getenv("OPENAI_API_KEY"))
    parser = argparse.ArgumentParser(description="Simple RAG with LangChain + Chroma + OpenAI")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing PDFs and text files to ingest")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Directory to persist Chroma DB")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into Chroma")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for text splitter")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for text splitter")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for chat responses")

    args = parser.parse_args()

    
    if args.ingest:
        ingest(args.data_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)
    else:
        start_chat(args.persist_dir, temperature=args.temperature)


if __name__ == "__main__":
    main()
