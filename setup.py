from setuptools import setup, find_packages

setup(
    name="hybrid-dense-reranker",
    version="1.0.0",
    description="A hybrid dense reranker using TF-IDF embeddings and Anthropic Claude for intelligent reranking",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "numpy>=1.21.0",
        "faiss-cpu>=1.7.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "anthropic>=0.7.0",
        "scikit-learn>=1.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)