# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAG application combining TF-IDF embeddings with Claude for document reranking. Scoring: 30% TF-IDF + 70% Claude relevance.

**Project Structure:**
```
src/hybrid_dense_reranker/  # Main application code
tests/                      # All test files
data/                       # Corpus data files
main.py                     # Entry point
```

## Quick Start

```bash
# Setup
./setup_venv.sh  # or setup_venv.bat on Windows
python main.py

# Access UI
# Open browser to: http://localhost:5002

# Test
python tests/validate_test_setup.py
python tests/run_integration_tests.py
```

## Essential Commands

```bash
# Development
pip install -r requirements.txt
python main.py                    # Run the application
pip install -e .                  # Install in development mode
hybrid-dense-reranker            # Run via console script

# Testing
pytest                           # Run all tests with coverage
pytest tests/test_integration.py # Run specific test file
pytest -m "not slow"            # Skip slow tests
python tests/test_corpus_quick.py # Quick corpus validation

# Test dependencies
pip install -r test_requirements.txt
```

## Architecture

**Modular Structure:**
- **src/hybrid_dense_reranker/app.py**: Flask app with web UI and API endpoints
- **src/hybrid_dense_reranker/search.py**: HybridReranker class for search logic
- **src/hybrid_dense_reranker/embeddings.py**: EmbeddingManager for TF-IDF
- **src/hybrid_dense_reranker/claude_client.py**: Claude API integration
- **src/hybrid_dense_reranker/corpus.py**: Corpus loading and management
- **src/hybrid_dense_reranker/config.py**: Configuration management
- **src/hybrid_dense_reranker/templates/**: HTML templates for web UI
- **src/hybrid_dense_reranker/static/**: CSS and JavaScript files

## Environment Variables (.env)

```bash
ANTHROPIC_API_KEY=your-key
CORPUS_SOURCE=default      # or 'mormon'
CHUNK_SIZE=1000           # Mormon corpus chunking
CHUNK_OVERLAP=100
FLASK_PORT=5002           # Web server port (default: 5002)
```

## Key Implementation Notes

1. **Enhanced Accuracy**: Advanced text preprocessing, query expansion, multi-factor scoring
2. **Advanced Embeddings**: TF-IDF with LSA, n-grams (1-3), 5000 features
3. **Smart Claude Prompting**: Detailed reasoning framework, contextual analysis
4. **Semantic Matching**: Keyword extraction, domain-aware preprocessing
5. **Testing**: Real API calls, no mocking. Check `pytest.ini` for coverage config
6. **Performance**: Embeddings/index built once at startup, retrieves k*2 for reranking

## Common Tasks

- **Access Web UI**: Navigate to `http://localhost:5002` after starting the app
- **Switch corpus**: Change `CORPUS_SOURCE` in .env and restart
- **Run specific tests**: `pytest tests/test_integration.py::TestAppIntegration -v`
- **View coverage**: Open `htmlcov/index.html` after running pytest
- **Validate setup**: Always run `python tests/validate_test_setup.py` first
- **Add new features**: Create modules in `src/hybrid_dense_reranker/` and update imports
- **Health check**: Visit `/health-page` in browser or GET `/health` for JSON

## UI Features

- **Interactive Search**: Clean web interface with real-time search
- **Enhanced Score Breakdown**: Shows TF-IDF, Claude, Semantic, and combined scores
- **Relevance Explanations**: AI-generated explanations for each result
- **Example Queries**: Pre-built queries for different corpus types
- **Responsive Design**: Works on desktop and mobile devices
- **Health Dashboard**: System status and corpus information

## RAG Accuracy Improvements

**Enhanced Embeddings:**
- TF-IDF with 5000 features (vs. 1000 previously)
- N-gram support (1-3) for better phrase matching
- LSA dimensionality reduction for semantic understanding
- Advanced text preprocessing with domain-specific awareness

**Smart Query Processing:**
- Query expansion with synonyms and related terms
- Keyword extraction with stemming and domain preservation
- Semantic similarity calculation using Jaccard + domain boosting

**Advanced Claude Integration:**
- Structured prompting with detailed scoring criteria
- Contextual analysis considering other retrieved documents
- Latest Claude-3.5-Sonnet model for better understanding
- Consistent scoring with low temperature settings

**Multi-Factor Scoring:**
- TF-IDF (20%), Claude (50%), Semantic (20%), Length (5%), Keywords (5%)
- Normalized scoring across all components
- Keyword density bonuses for domain term matches
- Query-content length appropriateness factors

**Chunk Ordering Fix:**
- **Sequential Reordering**: Mormon corpus chunks automatically sorted by chunk_id for narrative flow
- **Dual Ordering Options**: Choose between "sequential" (narrative order) or "relevance" (best scores first)
- **Smart Separation**: Mormon chunks ordered sequentially, other content by relevance
- **UI Controls**: Dropdown to select ordering preference
- **Context API**: New `/chunk-context/<id>` endpoint for narrative context
- **Metadata Tracking**: Maintains chunk_id, source, and corpus_index for all results
```
</invoke>