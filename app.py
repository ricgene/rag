import streamlit as st
from rag_app import load_documents, split_documents, create_vector_store, setup_rag_chain

# Set page config
st.set_page_config(page_title="Financial Document RAG System", page_icon="📊")

# Title
st.title("Financial Document RAG System")

# Load documents and set up RAG chain
@st.cache(allow_output_mutation=True)
def load_rag_chain():
    docs = load_documents()
    splits = split_documents(docs)
    vs = create_vector_store(splits)
    return setup_rag_chain(vs)

# Load the RAG chain
rag_chain = load_rag_chain()

# User input
user_question = st.text_input("Ask a question about the financial documents:")

# Process the question
if user_question:
    with st.spinner("Searching documents and generating answer..."):
        answer = rag_chain.invoke(user_question)
        st.write("Answer:", answer)

# Instructions
st.markdown("""
### Instructions
- Enter your question in the text input above.
- The system will search through the financial documents and provide an answer based on the available data.
""") 