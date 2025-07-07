from flask import Flask, request, jsonify
import numpy as np
import faiss
import requests
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Constants
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Configuration
CORPUS_SOURCE = os.getenv("CORPUS_SOURCE", "default")  # "default" or "mormon"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlap between chunks

def load_mormon_corpus():
    """Load and chunk the Mormon text from the data file."""
    try:
        with open('data/mormon13short.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into verses - the format is "1 Nephi 1:1" followed by verse number and content
        verses = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for verse lines that start with a number and contain actual verse content
            # Format: " 1 I, Nephi, having been born of goodly parents..."
            if re.match(r'^\s*\d+\s+[A-Z]', line) and len(line) > 30:
                # Extract the verse content (everything after the verse number)
                verse_match = re.search(r'^\s*\d+\s+(.+)', line)
                if verse_match:
                    verse_content = verse_match.group(1).strip()
                    if verse_content and len(verse_content) > 20:  # Filter out very short lines
                        verses.append(verse_content)
        
        # If no verses found with the above pattern, try a more general approach
        if len(verses) == 0:
            print("No verses found with standard pattern, trying alternative parsing...")
            for line in lines:
                line = line.strip()
                # Look for any line that seems to contain substantial text content
                if (len(line) > 50 and
                    not line.startswith('*') and
                    not line.startswith('[') and
                    not line.startswith('Chapter') and
                    not re.match(r'^1 Nephi \d+$', line) and
                    not line.isupper() and
                    'Nephi' in line or 'Lord' in line or 'came to pass' in line):
                    verses.append(line)
        
        # Create chunks from verses
        corpus = []
        current_chunk = ""
        chunk_id = 1
        
        for verse in verses:
            # If adding this verse would exceed chunk size, save current chunk and start new one
            if len(current_chunk) + len(verse) + 1 > CHUNK_SIZE and current_chunk:
                corpus.append({
                    "title": f"Book of Mormon - Chunk {chunk_id}",
                    "content": current_chunk.strip()
                })
                chunk_id += 1
                # Start new chunk with overlap
                if CHUNK_OVERLAP > 0 and len(current_chunk) > CHUNK_OVERLAP:
                    current_chunk = current_chunk[-CHUNK_OVERLAP:] + " " + verse
                else:
                    current_chunk = verse
            else:
                # Add verse to current chunk
                if current_chunk:
                    current_chunk += " " + verse
                else:
                    current_chunk = verse
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            corpus.append({
                "title": f"Book of Mormon - Chunk {chunk_id}",
                "content": current_chunk.strip()
            })
        
        print(f"Loaded {len(corpus)} chunks from Mormon text (parsed {len(verses)} verses)")
        
        # If we still have no corpus, fall back to default
        if len(corpus) == 0:
            print("No content could be parsed from Mormon text, falling back to default corpus")
            return get_default_corpus()
            
        return corpus
        
    except FileNotFoundError:
        print("Mormon text file not found, falling back to default corpus")
        return get_default_corpus()
    except Exception as e:
        print(f"Error loading Mormon corpus: {e}, falling back to default corpus")
        return get_default_corpus()

def get_default_corpus():
    """Return the default sample corpus."""
    return [
        {"title": "Legal Risk Report - 2023", "content": "The contract exposes the organization to liability due to lack of indemnification clauses."},
        {"title": "Security Memo", "content": "Ensure all employees use 2FA to reduce unauthorized access risks."},
        {"title": "Financial Summary", "content": "Revenue grew by 15% but legal expenses increased due to ongoing litigation."}
    ]

def load_corpus():
    """Load the corpus based on configuration."""
    if CORPUS_SOURCE.lower() == "mormon":
        return load_mormon_corpus()
    else:
        return get_default_corpus()

# Load the corpus
corpus = load_corpus()

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
