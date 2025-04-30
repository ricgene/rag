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

