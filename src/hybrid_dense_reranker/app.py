"""Flask application for the Hybrid Dense Reranker."""
from flask import Flask, request, jsonify, render_template
from .search import HybridReranker
from .config import FLASK_DEBUG, FLASK_PORT


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Initialize the hybrid reranker
    reranker = HybridReranker()
    
    @app.route("/")
    def index():
        """Main search interface."""
        return render_template("index.html")
    
    @app.route("/health-page")
    def health_page():
        """Health status page."""
        return render_template("health.html")
    
    @app.route("/rag-query", methods=["POST"])
    def rag_query():
        """Handle RAG query requests."""
        try:
            data = request.json
            if not data or "query" not in data:
                return jsonify({"error": "Missing 'query' field in request"}), 400
            
            query = data.get("query")
            if not query or not query.strip():
                return jsonify({"error": "Query cannot be empty"}), 400
            
            # Use simplified search with sequential ordering
            search_response = reranker.search_with_context(query, k=5, include_context=True)
            results = search_response["results"]
            
            # Return simplified response
            response_data = {
                "results": results,
                "has_sequential_content": search_response.get("has_sequential_content", False),
                "context_available": bool(search_response.get("context", {}))
            }
            
            return jsonify(response_data)
        
        except Exception as e:
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "corpus_size": len(reranker.corpus)})
    
    @app.route("/chunk-context/<int:chunk_id>", methods=["GET"])
    def get_chunk_context(chunk_id):
        """Get context around a specific chunk."""
        try:
            context_chunks = reranker.get_chunk_context(chunk_id, "mormon", context_size=3)
            return jsonify({
                "chunk_id": chunk_id,
                "context": context_chunks,
                "total_context_chunks": len(context_chunks)
            })
        except Exception as e:
            return jsonify({"error": f"Error getting context: {str(e)}"}), 500
    
    return app


def run_app():
    """Run the Flask application."""
    app = create_app()
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)