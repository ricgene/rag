import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import json
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import glob
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re

# Load environment variables from .env file in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Get OpenAI API key from environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file or Streamlit secrets")

# Define the directory containing your contract markdown files
contracts_dir = "financial_docs/"

def clean_html_to_text(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    return text

# Step 1: Load documents
def load_documents() -> List[Document]:
    """Load and preprocess documents from the financial_docs directory."""
    documents = []
    
    # Process PDF files
    pdf_loader = DirectoryLoader(
        "financial_docs/",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        # Extract company and document type from filename
        filename = os.path.basename(doc.metadata["source"])
        company = "Unknown"
        doc_type = "Unknown"
        
        # Try to extract company name from filename
        if "CIENA" in filename.upper():
            company = "Ciena"
        elif "NOKIA" in filename.upper():
            company = "Nokia"
        elif "INFINERA" in filename.upper():
            company = "Infinera"
        elif "CISCO" in filename.upper():
            company = "Cisco"
            
        # Try to determine document type
        if "EARNINGS" in filename.upper() or "Q1" in filename.upper() or "Q2" in filename.upper() or "Q3" in filename.upper() or "Q4" in filename.upper():
            doc_type = "Earnings Report"
        elif "TRANSCRIPT" in filename.upper():
            doc_type = "Earnings Call Transcript"
        elif "PRESENTATION" in filename.upper():
            doc_type = "Investor Presentation"
        elif "PRESS" in filename.upper() or "RELEASE" in filename.upper():
            doc_type = "Press Release"
            
        # Add metadata
        doc.metadata.update({
            "company": company,
            "document_type": doc_type,
            "year": "2025" if "2025" in filename else "2024" if "2024" in filename else "Unknown",
            "quarter": "Q1" if "Q1" in filename else "Q2" if "Q2" in filename else "Q3" if "Q3" in filename else "Q4" if "Q4" in filename else "Unknown",
            "filename": filename,
            "file_type": "PDF"
        })
        documents.append(doc)
    
    # Process Excel files (split by sheet)
    for xlsx_path in glob.glob("financial_docs/*.xlsx"):
        filename = os.path.basename(xlsx_path)
        company = "Unknown"
        doc_type = "Unknown"
        # Try to extract company name from filename
        if "CIENA" in filename.upper():
            company = "Ciena"
        elif "NOKIA" in filename.upper():
            company = "Nokia"
        elif "INFINERA" in filename.upper():
            company = "Infinera"
        elif "CISCO" in filename.upper():
            company = "Cisco"
        # Try to determine document type
        if "TABLES" in filename.upper():
            doc_type = "Financial Tables"
        elif "RESULTS" in filename.upper():
            doc_type = "Financial Results"
        # Load each sheet as a separate document
        try:
            xls = pd.ExcelFile(xlsx_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                text = df.to_csv(index=False)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": xlsx_path,
                        "company": company,
                        "document_type": doc_type,
                        "year": "2025" if "2025" in filename else "2024" if "2024" in filename else "Unknown",
                        "quarter": "Q1" if "Q1" in filename else "Q2" if "Q2" in filename else "Q3" if "Q3" in filename else "Q4" if "Q4" in filename else "Unknown",
                        "filename": filename,
                        "file_type": "Excel",
                        "sheet_name": sheet_name
                    }
                )
                documents.append(doc)
                print(f"Loaded Excel file: {filename}, sheet: {sheet_name}, rows: {len(df)}")
        except Exception as e:
            print(f"Error processing Excel file {filename}: {e}")
    
    # Process HTML files
    html_docs = []
    for html_path in glob.glob("financial_docs/*.html"):
        try:
            text = clean_html_to_text(html_path)
            filename = os.path.basename(html_path)
            company = "Unknown"
            doc_type = "Unknown"
            
            # Try to extract company name from filename
            if "CIENA" in filename.upper():
                company = "Ciena"
            elif "NOKIA" in filename.upper():
                company = "Nokia"
            elif "INFINERA" in filename.upper():
                company = "Infinera"
            elif "CISCO" in filename.upper():
                company = "Cisco"
                
            # Try to determine document type
            if "NEWS" in filename.upper():
                doc_type = "News Article"
            elif "PRESS" in filename.upper():
                doc_type = "Press Release"
                
            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": html_path,
                    "company": company,
                    "document_type": doc_type,
                    "year": "2025" if "2025" in filename else "2024" if "2024" in filename else "Unknown",
                    "quarter": "Q1" if "Q1" in filename else "Q2" if "Q2" in filename else "Q3" if "Q3" in filename else "Q4" if "Q4" in filename else "Unknown",
                    "filename": filename,
                    "file_type": "HTML"
                }
            )
            html_docs.append(doc)
        except Exception as e:
            print(f"Error processing {html_path}: {e}")
    
    documents.extend(html_docs)
    print(f"Loaded {len(documents)} documents with metadata")
    return documents

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
def setup_rag_chain(vector_store: FAISS) -> RunnablePassthrough:
    """Set up the RAG chain with the vector store."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Custom prompt template that includes metadata
    template = """You are a financial document analysis assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    For each piece of context, you have the following metadata:
    - Company: The company the document is about
    - Document Type: The type of document (e.g., Earnings Report, Press Release)
    - Year: The year the document is from
    - Quarter: The quarter the document is from (if applicable)
    - Filename: The name of the file
    - File Type: The type of file (e.g., PDF, Excel, HTML)
    - Sheet Name: The name of the sheet (if applicable)

    Context:
    {context}

    Question: {question}

    Instructions:
    1. First, identify which companies are mentioned in the context and their document types.
    2. For each company, summarize any financial data or key information available.
    3. If the question is about a specific company, focus on that company's data.
    4. If the question is general, provide a comprehensive answer across all companies.
    5. Always cite the source of information (company name and document type).
    6. If no relevant information is found, say so explicitly.
    7. If the document type is 'Unknown', respond with: 'I apologize, I do not have the documents to answer your question.'
    8. If the question is about tabular data, or if any document has file type 'Excel', 'CSV', or a sheet name, always list these documents, mentioning the company, filename, and sheet name (if available), even if the question is general.

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    
    return chain

    
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