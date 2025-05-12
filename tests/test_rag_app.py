import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_app import load_documents, split_documents, create_vector_store, setup_rag_chain

# Test fixtures
@pytest.fixture
def mock_documents():
    return [
        Mock(page_content="Test document 1", metadata={"source": "test1.pdf"}),
        Mock(page_content="Test document 2", metadata={"source": "test2.pdf"})
    ]

@pytest.fixture
def mock_splits():
    return [
        Mock(page_content="Test chunk 1", metadata={"source": "test1.pdf"}),
        Mock(page_content="Test chunk 2", metadata={"source": "test1.pdf"}),
        Mock(page_content="Test chunk 3", metadata={"source": "test2.pdf"})
    ]

# Test environment setup
def test_environment_variables():
    """Test that required environment variables are set."""
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY environment variable is not set"

# Test document loading
@patch('rag_app.DirectoryLoader')
def test_load_documents(mock_loader, mock_documents):
    """Test document loading functionality."""
    mock_loader.return_value.load.return_value = mock_documents
    documents = load_documents()
    assert len(documents) == 2
    assert documents[0].page_content == "Test document 1"
    assert documents[1].page_content == "Test document 2"

# Test document splitting
def test_split_documents(mock_documents):
    """Test document splitting functionality."""
    splits = split_documents(mock_documents)
    assert len(splits) > 0
    assert all(hasattr(split, 'page_content') for split in splits)

# Test vector store creation
@patch('rag_app.OpenAIEmbeddings')
@patch('rag_app.FAISS')
def test_create_vector_store(mock_faiss, mock_embeddings, mock_splits):
    """Test vector store creation functionality."""
    mock_faiss.from_documents.return_value = Mock()
    vector_store = create_vector_store(mock_splits)
    assert vector_store is not None
    mock_faiss.from_documents.assert_called_once()

# Test RAG chain setup
def test_setup_rag_chain():
    """Test RAG chain setup functionality with dependency injection."""
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage

    # Create a real RunnablePassthrough as the retriever
    mock_vector_store = Mock()
    mock_vector_store.as_retriever.return_value = RunnablePassthrough()

    # Create real LangChain objects for testing
    template = "Test template {context} {question}"
    prompt = ChatPromptTemplate.from_template(template)

    # Create a proper mock LLM that implements BaseChatModel
    class MockLLM(BaseChatModel):
        @property
        def _llm_type(self):
            return "mock"

        def _generate(self, messages, stop=None):
            return {"generations": [{"message": HumanMessage(content="Test response")}]}

    # Set up the RAG chain
    rag_chain = setup_rag_chain(mock_vector_store, llm=MockLLM(), prompt=prompt)
    assert rag_chain is not None

# Integration test
@pytest.mark.integration
def test_end_to_end_rag():
    """Test the complete RAG pipeline with a simple question."""
    # Skip if no API key is set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Create a test PDF document
    from fpdf import FPDF
    os.makedirs("financial_docs", exist_ok=True)
    pdf_path = "financial_docs/test_doc.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test document about artificial intelligence.", ln=True)
    pdf.output(pdf_path)
    
    try:
        # Load and process the document
        documents = load_documents()
        splits = split_documents(documents)
        vector_store = create_vector_store(splits)
        rag_chain = setup_rag_chain(vector_store)
        
        # Test a simple question
        question = "What is this document about?"
        answer = rag_chain.invoke(question)
        
        assert answer is not None
        assert len(answer) > 0
    finally:
        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists("telecom_faiss_index"):
            import shutil
            shutil.rmtree("telecom_faiss_index") 