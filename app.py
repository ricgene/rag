import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from rag_app import load_documents, split_documents, create_vector_store, setup_rag_chain

# Debug: Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Load environment variables from .env file in current directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {env_path}")
load_dotenv(env_path)

# Debug: Print the API key (first few characters only)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"API Key loaded (first 4 chars): {api_key[:4]}...")
    print(f"API Key length: {len(api_key)}")
else:
    print("No API key found in environment")

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set your OpenAI API key in the environment variables or .env file")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Financial Document RAG System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Initialize session state for vector store and document info
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'is_reindexed' not in st.session_state:
    st.session_state.is_reindexed = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'answer' not in st.session_state:
    st.session_state.answer = None

# Try to load existing vector store from disk
vector_store_path = "vector_store/faiss_index"
index_file = os.path.join(vector_store_path, "index.faiss")
if os.path.exists(index_file) and st.session_state.vector_store is None:
    print("Loading existing vector store from disk...")
    try:
        embeddings = OpenAIEmbeddings()
        try:
            # Try with allow_dangerous_deserialization for newer versions
            st.session_state.vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except TypeError:
            # Fall back to older version without the parameter
            st.session_state.vector_store = FAISS.load_local(vector_store_path, embeddings)
        st.session_state.file_name = "All Documents"
        st.session_state.is_reindexed = True
        print("Successfully loaded existing vector store")
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        st.error(f"Failed to load vector store: {str(e)}")
        st.info("Try reindexing your documents using the 'Reindex All Documents' button in the sidebar.")
        st.session_state.vector_store = None

# Password check
if not st.session_state.authenticated:
    st.title("Financial Document RAG System")
    # Add specific styling for password input
    st.markdown("""
    <style>
        .stTextInput input[type="password"] {
            border: 2px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
            display: block;
        }
        .stTextInput input[type="password"]:focus {
            border-color: #4CAF50;
            outline: none;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)
    password = st.text_input("Enter password to access the application:", type="password", key="login_password")
    if st.button("Login"):
        # Using environment variable for password, defaulting to 'onion' if not set
        correct_password = os.getenv("APP_PASSWORD", "onion")
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

# Add custom CSS for file uploader (only after authentication)
st.markdown("""
<style>
    /* Style for the file uploader container */
    .stFileUploader {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* Hover effect */
    .stFileUploader:hover {
        border-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Style when file is selected */
    .stFileUploader[data-has-file="true"] {
        border-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Style for the upload button */
    .stFileUploader button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .stFileUploader button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Main app content (only shown after authentication)
st.title("Financial Document RAG System")
st.markdown("""
This application allows you to ask questions about financial documents using AI.
The system processes PDF, Excel, and HTML files to provide accurate answers based on the document content.

⚠️ **Note**: This is a public demo. Please do not submit sensitive or confidential information.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    # Move file uploader to sidebar
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "xlsx", "xls", "html", "htm"])
    
    model_name = st.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
        index=0,
        help="Select the OpenAI model to use"
    )
    chunk_size = st.slider(
        "Chunk Size", 
        min_value=500, 
        max_value=2000, 
        value=1000,
        help="Size of document chunks in characters"
    )
    k_value = st.slider(
        "Number of chunks to retrieve (k)", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Number of most relevant chunks to retrieve per query"
    )
    
    st.markdown("---")
    st.markdown("### Document Management")
    if st.button("Reindex All Documents"):
        with st.spinner("Reindexing all documents..."):
            try:
                # Clear existing vector store
                st.session_state.vector_store = None
                if 'rag_chain' in st.session_state:
                    del st.session_state.rag_chain
                
                # Load and process all documents
                documents = load_documents()
                if documents:
                    splits = split_documents(documents)
                    vector_store = create_vector_store(splits)
                    st.session_state.vector_store = vector_store
                    st.session_state.file_name = "All Documents"
                    st.session_state.is_reindexed = True
                    st.success(f"Successfully reindexed {len(documents)} documents")
                    st.rerun()  # Force a rerun to update the UI
                else:
                    st.warning("No documents found in the financial_docs directory")
            except Exception as e:
                st.error(f"Error reindexing documents: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### Disclaimer
    This is a demo application. The responses are based on the provided financial documents and may not be complete or up-to-date.
    """)
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Document processing functions
def extract_company_info(doc_content, filename):
    """Extract company information from document content and filename."""
    # Common company indicators in financial documents
    company_indicators = [
        "About", "Company Overview", "Corporate Profile",
        "About Us", "Company Information", "Corporate Information"
    ]
    
    # Try to find company name in the first few paragraphs
    content_lower = doc_content.lower()
    first_paragraphs = doc_content.split('\n\n')[:3]  # Look at first 3 paragraphs
    
    # First try to find company name in the first few paragraphs
    for para in first_paragraphs:
        para_lower = para.lower()
        # Look for common company introduction patterns
        for indicator in company_indicators:
            if indicator.lower() in para_lower:
                # Try to extract the company name from the next sentence
                sentences = para.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Avoid very short sentences
                        return sentence.strip()
    
    # If not found in paragraphs, try to extract from filename
    filename_upper = filename.upper()
    # Look for common company name patterns in filename
    words = filename_upper.split()
    for word in words:
        if len(word) > 3:  # Avoid very short words
            # Clean the word (remove special characters)
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 3:
                return clean_word.title()
    
    return "Unknown"

def process_document(uploaded_file, existing_vector_store=None):
    """Process an uploaded document based on its file type."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            file_path = temp_file.name
        
        # Extract metadata from filename
        filename = uploaded_file.name
        doc_type = "Unknown"
        
        # Try to determine document type
        if "EARNINGS" in filename.upper() or any(q in filename.upper() for q in ["Q1", "Q2", "Q3", "Q4"]):
            doc_type = "Earnings Report"
        elif "TRANSCRIPT" in filename.upper():
            doc_type = "Earnings Call Transcript"
        
        # Load document based on file type
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(file_path)
        elif filename.lower().endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Load and process the document
        documents = loader.load()
        
        # Add metadata to documents
        for doc in documents:
            doc.metadata.update({
                "source": filename,
                "doc_type": doc_type,
                "company": extract_company_info(doc.page_content, filename)
            })
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get('chunk_size', 1000),
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # If there's an existing vector store, merge with it
        if existing_vector_store:
            existing_vector_store.add_documents(splits)
            vector_store = existing_vector_store
        else:
            vector_store = FAISS.from_documents(splits, embeddings)
        
        # Clean up temporary file
        os.unlink(file_path)
        
        return vector_store, filename
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        if 'file_path' in locals():
            os.unlink(file_path)
        return None, None

def setup_rag_chain(vector_store):
    """Set up the RAG chain for querying."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
    
    template = """You are an AI assistant specialized in analyzing financial documents and investment materials.

Answer the question based ONLY on the following context:
{context}

Question: {question}

Your answer should:
1. Be specific and directly reference information from the documents
2. Extract and highlight financial metrics when relevant
3. Be well-structured and easy to read
4. Only state what is directly supported by the context

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(f"DOCUMENT SECTION {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main app interface
# Always show the question input field
st.write("### Ask a Question")
question = st.text_input("Enter your question:", value=st.session_state.get("question", ""))

# Clear answer if question changes
if question != st.session_state.last_question:
    st.session_state.answer = None
    st.session_state.last_question = question

# Generate answer if question is provided
if question:
    print("\n=== DEBUG: Before generating answer ===")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"API Key before answer generation (first 4 chars): {api_key[:4]}...")
        print(f"API Key length before answer generation: {len(api_key)}")
    else:
        print("No API key found in environment before answer generation")
    print("=== End API Key Check ===\n")
    
    with st.spinner("Generating answer..."):
        try:
            start_time = time.time()
            # Create RAG chain only once and reuse
            if 'rag_chain' not in st.session_state:
                st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
            answer = st.session_state.rag_chain.invoke(question)
            processing_time = time.time() - start_time
            
            st.write("### Answer")
            st.write(answer)
            st.info(f"Processing time: {processing_time:.2f} seconds")
            
            if "question" in st.session_state:
                del st.session_state.question
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.info("Try reindexing the documents or uploading a new document.")

# Display example questions if a vector store exists
if st.session_state.vector_store is not None:
    st.write(f"### Ask questions about {st.session_state.file_name}")
    
    # Example questions
    st.write("Try questions like:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What are the key financial metrics?"):
            st.session_state.question = "What are the key financial metrics mentioned in the document?"
    with col2:
        if st.button("What are the main risk factors?"):
            st.session_state.question = "What are the main risk factors or challenges mentioned in the document?"

# Process the uploaded file
if uploaded_file:
    with st.spinner("Processing document..."):
        vector_store, filename = process_document(uploaded_file, st.session_state.get('vector_store'))
        if vector_store:
            st.session_state.vector_store = vector_store
            if st.session_state.file_name == "All Documents":
                st.session_state.file_name = f"All Documents + {filename}"
            else:
                st.session_state.file_name = filename
            st.session_state.is_reindexed = False
            # Clear the RAG chain when new document is uploaded
            if 'rag_chain' in st.session_state:
                del st.session_state.rag_chain
            st.success(f"Successfully processed {filename}")

# No file uploaded yet
else:
    st.info("Please upload a document or reindex existing documents to begin analysis")

# Instructions
st.markdown("""
### Instructions
- Upload a document or use the "Reindex All Documents" button to process documents
- Enter your question in the text input above
- The system will search through the documents and provide an answer based on the available data
""") 