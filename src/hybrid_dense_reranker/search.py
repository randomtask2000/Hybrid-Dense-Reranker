"""Enhanced search and ranking functionality with advanced algorithms."""
import numpy as np
import faiss
from typing import List, Dict, Tuple
from .embeddings import EmbeddingManager
from .claude_client import ClaudeClient
from .corpus import load_corpus
from .text_processor import AdvancedTextProcessor


class HybridReranker:
    """Advanced hybrid search and reranking system with enhanced accuracy."""
    
    def __init__(self):
        """Initialize the enhanced hybrid reranker."""
        self.embedding_manager = EmbeddingManager(max_features=5000, use_lsa=True)
        self.claude_client = ClaudeClient()
        self.text_processor = AdvancedTextProcessor()
        self.corpus = load_corpus()
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """Build the FAISS index with TF-IDF embeddings."""
        texts = [doc["content"] for doc in self.corpus]
        
        # Fit vectorizer on all texts first
        self.embedding_manager.fit(texts)
        
        # Generate embeddings for all documents
        doc_embeddings = self.embedding_manager.get_embeddings_batch(texts)
        
        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced hybrid search with query expansion and advanced scoring."""
        # Expand query for better matching
        expanded_query = self.text_processor.expand_query(query)
        
        # Get query embedding for expanded query
        query_embedding = np.array(self.embedding_manager.get_embedding(expanded_query)).astype("float32")
        
        # Retrieve more candidates for better reranking (k*2)
        retrieve_k = min(k * 2, len(self.corpus))
        D, I = self.index.search(np.array([query_embedding]), k=retrieve_k)
        
        # Get retrieved documents
        retrieved = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.corpus):  # Ensure valid index
                doc = self.corpus[idx]
                tfidf_score = float(D[0][i])
                retrieved.append({
                    "doc": doc,
                    "tfidf_score": tfidf_score,
                    "index": idx
                })
        
        # Calculate multiple relevance scores
        enhanced_results = []
        for item in retrieved:
            doc = item["doc"]
            tfidf_score = item["tfidf_score"]
            
            # Get Claude relevance score with context
            other_docs = [r["doc"] for r in retrieved if r["index"] != item["index"]][:2]
            claude_score = self.claude_client.analyze_relevance_with_context(
                doc["content"], query, other_docs
            )
            
            # Calculate semantic similarity
            semantic_score = self.embedding_manager.get_semantic_similarity(
                doc["content"], query
            )
            
            # Enhanced scoring with multiple factors
            combined_score = self._calculate_enhanced_score(
                tfidf_score, claude_score, semantic_score, doc["content"], query
            )
            
            enhanced_results.append({
                "title": doc["title"],
                "content": doc["content"],
                "tfidf_score": tfidf_score,
                "claude_score": claude_score,
                "semantic_score": semantic_score,
                "combined_score": combined_score,
                "explanation": self._generate_relevance_explanation(
                    doc["title"], query, claude_score, semantic_score
                )
            })
        
        # Sort by enhanced combined score and return top k
        enhanced_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add metadata to results
        for result in enhanced_results:
            if hasattr(self, 'corpus') and len(self.corpus) > 0:
                # Find the original document to get chunk metadata
                for i, doc in enumerate(self.corpus):
                    if (doc.get("title") == result["title"] and 
                        doc.get("content") == result["content"]):
                        result["corpus_index"] = i
                        result["chunk_id"] = doc.get("chunk_id", i + 1)
                        result["source"] = doc.get("source", "unknown")
                        break
        
        # Take top k results
        final_results = enhanced_results[:k]
        
        # Always sort chunks by sequential order (chunk_id) for narrative flow
        final_results.sort(key=lambda x: x.get("chunk_id", 999))
        
        return final_results
    
    def get_chunk_context(self, chunk_id: int, source: str, context_size: int = 2) -> List[Dict]:
        """Get surrounding chunks for better context understanding."""
        if source != "mormon":
            return []
        
        context_chunks = []
        for doc in self.corpus:
            if (doc.get("source") == "mormon" and 
                doc.get("chunk_id") is not None):
                doc_chunk_id = doc.get("chunk_id")
                # Get chunks within context_size of the target chunk
                if abs(doc_chunk_id - chunk_id) <= context_size:
                    context_chunks.append({
                        "chunk_id": doc_chunk_id,
                        "title": doc["title"],
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "distance": abs(doc_chunk_id - chunk_id)
                    })
        
        # Sort by chunk_id to maintain narrative order
        context_chunks.sort(key=lambda x: x["chunk_id"])
        return context_chunks
    
    def search_with_context(self, query: str, k: int = 5, include_context: bool = True) -> Dict:
        """Enhanced search that returns results in sequential order."""
        # Get search results (already sorted by chunk_id for sequential order)
        results = self.search(query, k)
        
        if not include_context:
            return {"results": results, "context": {}}
        
        # Add context information for Mormon corpus chunks
        context_info = {}
        for result in results:
            if result.get("source") == "mormon" and result.get("chunk_id"):
                chunk_id = result["chunk_id"]
                context_chunks = self.get_chunk_context(chunk_id, "mormon")
                context_info[chunk_id] = context_chunks
        
        return {
            "results": results,
            "context": context_info,
            "has_sequential_content": any(r.get("source") == "mormon" for r in results)
        }
    
    def _calculate_enhanced_score(self, tfidf_score: float, claude_score: float, 
                                semantic_score: float, content: str, query: str) -> float:
        """Calculate enhanced combined score with multiple factors."""
        # Normalize TF-IDF score (it can be > 1.0)
        normalized_tfidf = min(1.0, tfidf_score / max(1.0, np.max([1.0])))
        
        # Calculate query-content length ratio bonus
        query_words = len(query.split())
        content_words = len(content.split())
        length_ratio = min(1.0, query_words / max(1.0, content_words / 50))  # Normalize by expected chunk size
        
        # Calculate keyword density bonus
        query_keywords = self.text_processor.extract_keywords(query)
        content_keywords = self.text_processor.extract_keywords(content)
        keyword_overlap = len(query_keywords.intersection(content_keywords))
        keyword_density = keyword_overlap / max(1.0, len(query_keywords))
        
        # Weighted combination with enhanced factors
        weights = {
            'tfidf': 0.2,
            'claude': 0.5,      # Claude gets highest weight for semantic understanding
            'semantic': 0.2,    # Lexical semantic similarity
            'length': 0.05,     # Query-content length appropriateness
            'keywords': 0.05    # Keyword density bonus
        }
        
        combined_score = (
            weights['tfidf'] * normalized_tfidf +
            weights['claude'] * claude_score +
            weights['semantic'] * semantic_score +
            weights['length'] * length_ratio +
            weights['keywords'] * keyword_density
        )
        
        return min(1.0, combined_score)
    
    def _generate_relevance_explanation(self, title: str, query: str, 
                                      claude_score: float, semantic_score: float) -> str:
        """Generate human-readable explanation of relevance."""
        if claude_score >= 0.8 and semantic_score >= 0.6:
            return f"Highly relevant: '{title}' directly addresses '{query}' with strong semantic match."
        elif claude_score >= 0.6:
            return f"Very relevant: '{title}' contains substantial information related to '{query}'."
        elif claude_score >= 0.4 or semantic_score >= 0.4:
            return f"Moderately relevant: '{title}' has some useful information for '{query}'."
        else:
            return f"Limited relevance: '{title}' tangentially related to '{query}'."