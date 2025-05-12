from setuptools import setup, find_packages

setup(
    name="rag_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.25",
        "langchain-community>=0.3.24",
        "langchain-openai>=0.3.16",
        "python-dotenv>=1.1.0",
        "faiss-cpu>=1.11.0",
        "openai>=1.12.0",
        "pypdf>=4.0.1",
    ],
    extras_require={
        "test": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ],
    },
) 