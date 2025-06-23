# Hybrid Dense Reranker - Pure Anthropic

## Configuration

This application uses Anthropic's Claude for intelligent reranking and TF-IDF for embeddings, providing a pure Anthropic-based solution.

## Solution

Since Anthropic doesn't provide an embeddings API (they focus on text generation with Claude), this application combines:
- **TF-IDF embeddings** for initial document retrieval
- **Anthropic Claude** for intelligent relevance scoring and reranking

### Features:

1. **TF-IDF Embeddings**:
   - Fast, local embedding generation using scikit-learn
   - No external API calls for embeddings
   - Efficient for document retrieval

2. **Anthropic Claude Integration**:
   - Uses Claude-3-Sonnet for intelligent relevance scoring
   - Analyzes query-document relevance with natural language understanding
   - Combines TF-IDF and Claude scores for optimal results

3. **Hybrid Scoring**:
   - 30% TF-IDF similarity score
   - 70% Claude relevance score
   - Results sorted by combined score

## Setup Instructions

### 1. Install Dependencies

```bash
pip install flask numpy faiss-cpu requests python-dotenv anthropic scikit-learn
```

### 2. Set Environment Variable

You need an Anthropic API key for Claude. Get one from [Anthropic's Console](https://console.anthropic.com/).

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

To make it permanent, add it to your shell profile:

```bash
echo 'export ANTHROPIC_API_KEY="your-anthropic-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Test the Setup

Run the test script to verify everything works:

```bash
python test_embedding.py
```

### 4. Run the Application

```bash
python app.py
```

## Usage

The application provides a RAG (Retrieval-Augmented Generation) endpoint:

```bash
curl -X POST http://localhost:5000/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the security risks?"}'
```

## API Response Format

The application returns enhanced results with multiple scoring methods:
```json
[
  {
    "title": "Security Memo",
    "content": "Ensure all employees use 2FA to reduce unauthorized access risks.",
    "tfidf_score": 0.85,
    "claude_score": 0.92,
    "combined_score": 0.899
  }
]
```

## How It Works

1. **Initial Retrieval**: TF-IDF embeddings find potentially relevant documents
2. **Claude Analysis**: Each retrieved document is analyzed by Claude for relevance
3. **Hybrid Scoring**: Combines TF-IDF similarity with Claude's understanding
4. **Intelligent Ranking**: Results sorted by combined score for optimal relevance

## Benefits

- **Pure Anthropic**: Uses only Anthropic's Claude for AI processing
- **Cost Effective**: TF-IDF embeddings are free and fast
- **Intelligent**: Claude provides nuanced relevance understanding
- **Scalable**: Can handle large document collections efficiently