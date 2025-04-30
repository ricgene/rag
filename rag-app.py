import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Define the directory containing your contract markdown files
contracts_dir = "telecom_contracts/"

# Step 1: Load documents
def load_documents():
    print("Loading contract documents...")
    loader = DirectoryLoader(
        contracts_dir,
        glob="*.md",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
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
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    
    # Check if vector store exists locally
    if os.path.exists("telecom_faiss_index"):
        vector_store = FAISS.load_local("telecom_faiss_index", embeddings)
        print("Loaded existing vector store")
    else:
        vector_store = FAISS.from_documents(splits, embeddings)
        vector_store.save_local("telecom_faiss_index")
        print("Created and saved new vector store")
    
    return vector_store

# Step 4: Set up the RAG chain
def setup_rag_chain(vector_store):
    print("Setting up RAG chain...")
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Define the prompt template
    template = """You are an AI assistant specializing in telecommunications contracts and Service Level Agreements (SLAs).
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Your answer should be comprehensive, specific, and directly reference information from the contracts.
    If the information to answer the question is not in the provided context, say so clearly.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Set up the LLM
    llm = ChatOpenAI(model="gpt-4o")
    
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

# Main process
def main():
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

if __name__ == "__main__":
    main()