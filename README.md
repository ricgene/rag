# Install Python 3.11 if not already installed
# Then create a virtual environment
python3.11 -m venv venv-3.11

# Activate the virtual environment
source venv-3.11/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages using requirements.txt
pip install -r requirements.txt

# Run the script
python rag-app.py

# Deactivate the virtual environment
deactivate

# RAG app description
# Telecom Contract RAG System

## Project Overview
This project demonstrates a Retrieval Augmented Generation (RAG) system focused on telecommunications vendor contracts. It was created for an EY interview application to showcase RAG capabilities for extracting insights from contract documents.

## What is RAG?
RAG (Retrieval Augmented Generation) is a technique that enhances Large Language Models (LLMs) by providing them with relevant information retrieved from a knowledge base. This allows the model to generate responses based on specific documents rather than just its general training data.

## Features
- Load and process telecommunications contract documents
- Split documents into appropriate chunks for embedding
- Create vector embeddings of document chunks
- Store embeddings in a local vector database (FAISS)
- Query the contracts using natural language questions
- Generate comprehensive answers based on specific contract information

## Files in this Repository
- `main.py`: Main Python script for the RAG implementation
- `telecom_contracts/`: Directory containing the synthetic telecom contract documents
  - `network_sla_contract.md`: Network Performance SLA Contract
  - `equipment_maintenance_contract.md`: Equipment Maintenance Agreement
  - `technical_support_contract.md`: Technical Support SLA Contract
  - `incident_resolution_contract.md`: Incident Resolution SLA Contract
  - `penalties_contract.md`: Penalties for Non-Compliance Contract
- `telecom_faiss_index/`: Directory containing the FAISS vector database (created during first run)
- `requirements.txt`: Required Python packages

## Setup and Installation

### Prerequisites
- Python 3.8+ installed
- OpenAI API key

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/telecom-contract-rag.git
cd telecom-contract-rag
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key
Edit `main.py` and replace `your-openai-api-key-here` with your actual OpenAI API key.

### Running the Application
Execute the main Python script:
```bash
python main.py
```

The application will:
1. Load the contract documents
2. Split them into chunks
3. Create/load the vector database
4. Set up the RAG chain
5. Prompt you to enter questions about the contracts

## Example Questions
- "What are the network availability guarantees in the SLA?"
- "How is an incident classified as Critical severity?"
- "What penalties apply for missing response time targets?"
- "What is the escalation path for technical support issues?"
- "Under what conditions can a customer terminate a contract for poor service?"

## Implementation Details
- **Document Loading**: Uses LangChain's DirectoryLoader to load markdown files
- **Text Splitting**: Implements RecursiveCharacterTextSplitter with 1000-token chunks and 200-token overlap
- **Embeddings**: Uses OpenAI's text-embedding-ada-002 model via LangChain
- **Vector Store**: Implements FAISS, an efficient similarity search library
- **LLM**: Uses OpenAI's GPT-4o model for generating responses
- **Prompt Engineering**: Custom prompt template tailored for telecom contract analysis

## Why This Approach Works for the EY Application
- **Local Implementation**: Everything runs locally, requiring minimal setup
- **GitHub Integration**: Easy to showcase in a GitHub repository
- **Domain Relevance**: Directly addresses EY's current work with telecom clients
- **RAG Demonstration**: Shows full RAG implementation from document processing to query answering
- **No Cloud Requirements**: No need for complex cloud infrastructure, making it easy to demonstrate in an interview

## Future Enhancements
- Implement evaluation metrics to measure RAG performance
- Add document metadata for more refined searching
- Create a simple web interface for easier interaction
- Add multi-modal capabilities to handle contract diagrams and tables
- Implement hybrid search combining keyword and semantic searching

## Technologies Used
- LangChain: Framework for building LLM applications
- OpenAI: For embeddings and LLM capabilities
- FAISS: For efficient vector similarity search
- Python: Core programming language

# Financial Document RAG System

This is a Retrieval-Augmented Generation (RAG) system designed to analyze and answer questions about financial documents, including PDFs, Excel files, and HTML files. The system uses OpenAI's GPT models to provide intelligent responses based on the content of your documents.

## Running the Streamlit UI

To run the Streamlit UI for the RAG system, follow these steps:

1. Ensure you have activated your virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided in the terminal (usually http://localhost:8501).

4. Use the text input to ask questions about the financial documents.

## Features

- **Multi-Format Support**: Processes PDF, Excel (.xlsx), and HTML files
- **Local Vector Store**: Uses FAISS for efficient document storage and retrieval
- **Metadata Tracking**: Maintains information about processed documents
- **Interactive Q&A**: Ask questions about your financial documents in natural language
- **Document Chunking**: Intelligently splits documents for better context understanding

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd rag
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Project Structure

```
rag/
├── .venv/                  # Virtual environment
├── financial_docs/         # Directory for your documents
├── vector_store/          # Local vector database
├── rag_app/              # Main application code
│   └── __init__.py       # Core functionality
├── tests/                # Test suite
├── .env                  # Environment variables
└── requirements.txt      # Project dependencies
```

## Usage

1. **Prepare Your Documents**:
   - Place your PDF, Excel, and HTML files in the `financial_docs/` directory
   - Supported formats: `.pdf`, `.xlsx`, `.html`

2. **Run the Application**:
```bash
python -c "from rag_app import load_documents, split_documents, create_vector_store, setup_rag_chain; docs = load_documents(); splits = split_documents(docs); vs = create_vector_store(splits); chain = setup_rag_chain(vs); print(chain.invoke('Your question here?'))"
```

3. **Example Questions**:
   - "What is the forward guidance for the next quarter?"
   - "What were the GAAP results for Q1?"
   - "What are the key cash flow trends?"
   - "What is the revenue breakdown by segment?"

## How It Works

1. **Document Loading**:
   - The system loads documents from the `financial_docs/` directory
   - Supports PDF, Excel, and HTML files
   - Maintains document metadata and source information
   - HTML files are processed to extract text content while preserving structure

2. **Document Processing**:
   - Documents are split into manageable chunks
   - Each chunk is processed to maintain context
   - Default chunk size is 1000 characters with 200 character overlap

3. **Vector Store**:
   - Uses FAISS for efficient similarity search
   - Stores document embeddings locally
   - Maintains metadata about processed documents
   - Automatically updates when new documents are added

4. **Question Answering**:
   - Uses GPT-4 for generating responses
   - Retrieves relevant document chunks based on the question
   - Provides context-aware answers
   - Includes source information in responses

## What Happens When We Process Documents

1. **Document Ingestion**:
   - The system loads documents (PDFs, Excels, HTMLs) from the `financial_docs/` directory.
   - HTML files are preprocessed to extract clean text.

2. **Chunking**:
   - Documents are split into smaller chunks (e.g., 1000 characters with 200 character overlap) to facilitate processing.

3. **Vectorization**:
   - Each chunk is converted into a vector embedding using OpenAI's embedding model.
   - These embeddings are stored in a local FAISS vector store for efficient similarity search.

4. **Retrieval**:
   - When a question is asked, the system retrieves the most relevant chunks from the vector store based on similarity to the question.
   - The number of retrieved chunks (e.g., 8) is increased to capture more context.

5. **Prompting**:
   - The retrieved chunks are passed to the LLM (e.g., GPT-4) along with a prompt that instructs it to extract specific information (e.g., company names, financial data).
   - The prompt guides the model to provide a comprehensive answer based on the retrieved context.

6. **Response Generation**:
   - The LLM generates a response based on the retrieved context and the prompt instructions.
   - The response is returned to the user.

## Customization

You can customize various aspects of the system:

1. **Chunk Size**: Modify the `chunk_size` and `chunk_overlap` parameters in `split_documents()`
2. **Model**: Change the GPT model in `setup_rag_chain()`
3. **Prompt Template**: Customize the prompt template in `setup_rag_chain()`
4. **Retrieval Parameters**: Adjust the number of retrieved chunks in `setup_rag_chain()`

## Testing

Run the test suite:
```bash
pytest -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- OpenAI for GPT models
- LangChain for the RAG framework
- FAISS for vector storage

