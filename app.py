from flask import Flask, request, jsonify
import numpy as np
import faiss
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Constants
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Sample corpus
corpus = [
    {"title": "Legal Risk Report - 2023", "content": "The contract exposes the organization to liability due to lack of indemnification clauses."},
    {"title": "Security Memo", "content": "Ensure all employees use 2FA to reduce unauthorized access risks."},
    {"title": "Financial Summary", "content": "Revenue grew by 15% but legal expenses increased due to ongoing litigation."}
]

# Initialize TF-IDF vectorizer for embeddings (since Anthropic doesn't provide embeddings)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Helper function to get embeddings using TF-IDF
def get_embedding(text):
    """Create embeddings using TF-IDF since Anthropic doesn't provide embeddings API"""
    return vectorizer.transform([text]).toarray()[0]

# Helper function to use Anthropic Claude for text generation/analysis
def analyze_with_claude(text, query):
    """Use Anthropic Claude to analyze relevance between query and text"""
    try:
        message = ANTHROPIC_CLIENT.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": f"Rate the relevance of this text to the query on a scale of 0-1. Query: '{query}' Text: '{text}' Return only a number between 0 and 1."
                }
            ]
        )
        score = float(message.content[0].text.strip())
        return max(0, min(1, score))  # Ensure score is between 0 and 1
    except Exception as e:
        print(f"Claude analysis error: {e}")
        return 0.5  # Default score if Claude fails

# Build FAISS index with TF-IDF embeddings
texts = [doc["content"] for doc in corpus]
# Fit vectorizer on all texts first
vectorizer.fit(texts)
doc_embeddings = np.array([get_embedding(text) for text in texts]).astype("float32")
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

@app.route("/rag-query", methods=["POST"])
def rag_query():
    data = request.json
    query = data.get("query")

    # Get query embedding
    query_embedding = np.array(get_embedding(query)).astype("float32")

    # Dense retrieval using TF-IDF
    D, I = index.search(np.array([query_embedding]), k=3)
    retrieved = [corpus[i] for i in I[0]]

    # Enhanced scoring using Anthropic Claude
    results = []
    for i, doc in enumerate(retrieved):
        tfidf_score = float(D[0][i])
        claude_score = analyze_with_claude(doc["content"], query)
        # Combine TF-IDF and Claude scores
        combined_score = (tfidf_score * 0.3) + (claude_score * 0.7)
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "tfidf_score": tfidf_score,
            "claude_score": claude_score,
            "combined_score": combined_score
        })
    
    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
