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

# Title and description
st.title("Financial Document RAG System")
st.markdown("""
This application allows you to ask questions about financial documents using AI.
The system processes PDF, Excel, and HTML files to provide accurate answers based on the document content.
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