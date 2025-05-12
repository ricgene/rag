import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import json
from datetime import datetime
from typing import List

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Define the directory containing your contract markdown files
contracts_dir = "financial_docs/"

# Step 1: Load documents
def load_documents() -> List[Document]:
    """Load documents from the financial_docs directory."""
    loader = DirectoryLoader(
        "financial_docs",
        glob="**/*.*",
        loader_cls={
            "*.pdf": PyPDFLoader,
            "*.xlsx": UnstructuredExcelLoader,
            "*.html": UnstructuredHTMLLoader,
        },
    )
    return loader.load()

# Step 2: Split documents into chunks
def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    return splits

# Step 3: Create embeddings and store them
def create_vector_store(splits):
    """Create a vector store from document splits."""
    embeddings = OpenAIEmbeddings()
    vector_store_path = "vector_store/faiss_index"
    index_file = os.path.join(vector_store_path, "index.faiss")
    if os.path.exists(index_file):
        print("Attempting to load existing vector store...")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("Successfully loaded existing vector store")
    else:
        print("Creating new vector store...")
        vector_store = FAISS.from_documents(splits, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"Saved new vector store to {vector_store_path}")
        # Save metadata about the documents
        metadata = {
            "num_documents": len(splits),
            "created_at": datetime.now().isoformat(),
            "document_sources": list(set(doc.metadata.get("source", "unknown") for doc in splits))
        }
        with open(f"{vector_store_path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("Saved vector store metadata")
    return vector_store

# Step 4: Set up the RAG chain
def setup_rag_chain(vector_store, llm=None, prompt=None):
    print("Setting up RAG chain...")
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Allow injection of prompt and llm
    if prompt is None:
        template = """You are an AI assistant specializing in financial analysis and private equity.
        Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        Your answer should be comprehensive, specific, and directly reference information from the contracts.
        If the information to answer the question is not in the provided context, say so clearly.
        """
        prompt = ChatPromptTemplate.from_template(template)
    if llm is None:
        llm = ChatOpenAI(model="gpt-4")
    
    # Create a function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

    
def test_with_single_file(file_path):
    """Quick test with a single PDF file."""
    print(f"Testing with single file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splits = split_documents(documents)
    vector_store = create_vector_store(splits)
    rag_chain = setup_rag_chain(vector_store)
    
    print("\nRAG system is ready. You can now ask questions.")
    # Simple question loop
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        print("\nSearching documents and generating answer...")
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")

# Main process
def main2():
    # Process documents and create vector store
    documents = load_documents()
    splits = split_documents(documents)
    vector_store = create_vector_store(splits)
    rag_chain = setup_rag_chain(vector_store)
    
    print("\nRAG system is ready. You can now ask questions about the telecommunications contracts.")
    
    # Simple question loop
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        
        if question.lower() == 'exit':
            break
        
        print("\nSearching contracts and generating answer...")
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")

# Modify main to allow single file testing
def main():
    # Quick test with a single file option
    single_file = "sample_financial_doc.pdf"  # Replace with your file
    if os.path.exists(single_file):
        test_with_single_file(single_file)
        return
        
    # Original directory processing
    documents = load_documents()
    splits = split_documents(documents)
    vector_store = create_vector_store(splits)
    rag_chain = setup_rag_chain(vector_store)

if __name__ == "__main__":
    main()