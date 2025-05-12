from setuptools import setup, find_packages

setup(
    name="rag_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.13",
        "openai>=1.12.0",
        "python-dotenv>=1.0.0",
        "faiss-cpu>=1.7.4",
        "pandas>=2.2.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.1.0",
        "unstructured>=0.10.30",
        "pypdf>=4.0.1",
        "openpyxl>=3.1.2",
    ],
    python_requires=">=3.11",
) 