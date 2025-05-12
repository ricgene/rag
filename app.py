import streamlit as st
import os
from dotenv import load_dotenv
from rag_app import load_documents, split_documents, create_vector_store, setup_rag_chain

# Load environment variables
load_dotenv()

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

# Password check
if not st.session_state.authenticated:
    st.title("Financial Document RAG System")
    password = st.text_input("Enter password to access the application:", type="password")
    if password:
        # Using environment variable for password, defaulting to 'onion' if not set
        correct_password = os.getenv("APP_PASSWORD", "onion")
        if password == correct_password:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

# Main app content (only shown after authentication)
st.title("Financial Document RAG System")
st.markdown("""
This application allows you to ask questions about financial documents using AI.
The system processes PDF, Excel, and HTML files to provide accurate answers based on the document content.

⚠️ **Note**: This is a public demo. Please do not submit sensitive or confidential information.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system:
    - Processes multiple document formats
    - Uses OpenAI's GPT models
    - Provides context-aware answers
    - Maintains document metadata
    """)
    
    # Add a disclaimer
    st.markdown("---")
    st.markdown("""
    ### Disclaimer
    This is a demo application. The responses are based on the provided financial documents and may not be complete or up-to-date.
    """)
    
    # Add logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()

# Load documents and set up RAG chain
@st.cache_resource
def load_rag_chain():
    try:
        docs = load_documents()
        if not docs:
            st.warning("No documents found in the financial_docs directory")
            return None
        splits = split_documents(docs)
        vs = create_vector_store(splits)
        return setup_rag_chain(vs)
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

# Load the RAG chain
with st.spinner("Loading documents and setting up the RAG system..."):
    rag_chain = load_rag_chain()

if rag_chain:
    # User input
    user_question = st.text_input("Ask a question about the financial documents:")

    # Process the question
    if user_question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer = rag_chain.invoke(user_question)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    # Example questions
    st.markdown("""
    ### Example Questions
    - What is the forward guidance for the next quarter?
    - What were the GAAP results for Q1?
    - What are the key cash flow trends?
    - What is the revenue breakdown by segment?
    """)

# Instructions
st.markdown("""
### Instructions
- Enter your question in the text input above.
- The system will search through the financial documents and provide an answer based on the available data.
""") 