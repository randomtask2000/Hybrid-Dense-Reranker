"""Enhanced embedding generation using advanced TF-IDF."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from .text_processor import AdvancedTextProcessor


class EmbeddingManager:
    """Manages enhanced TF-IDF embeddings for the corpus."""
    
    def __init__(self, max_features=5000, use_lsa=True, lsa_components=300):
        """Initialize the enhanced TF-IDF vectorizer."""
        self.text_processor = AdvancedTextProcessor()
        self.use_lsa = use_lsa
        
        # Enhanced TF-IDF configuration
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            min_df=1,            # Include terms that appear in at least 1 document
            max_df=0.95,         # Exclude terms that appear in more than 95% of documents
            sublinear_tf=True,   # Apply sublinear scaling
            norm='l2'            # L2 normalization
        )
        
        # Optional LSA for dimensionality reduction and semantic enhancement
        if use_lsa:
            self.pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('lsa', TruncatedSVD(n_components=lsa_components, random_state=42)),
                ('normalizer', Normalizer(copy=False))
            ])
        else:
            self.pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('normalizer', Normalizer(copy=False))
            ])
        
        self.fitted = False
    
    def fit(self, texts):
        """Fit the pipeline on preprocessed texts."""
        # Preprocess all texts
        processed_texts = [self.text_processor.preprocess_for_embedding(text) for text in texts]
        
        # Fit the pipeline
        self.pipeline.fit(processed_texts)
        self.fitted = True
    
    def get_embedding(self, text):
        """Create enhanced embeddings using the preprocessing pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before generating embeddings")
        
        # Preprocess the text
        processed_text = self.text_processor.preprocess_for_embedding(text)
        
        # Transform using the pipeline
        embedding = self.pipeline.transform([processed_text])
        
        # Handle sparse vs dense matrices
        if hasattr(embedding, 'toarray'):
            return embedding.toarray()[0]
        else:
            return embedding[0]
    
    def get_embeddings_batch(self, texts):
        """Get embeddings for multiple texts at once."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before generating embeddings")
        
        # Preprocess all texts
        processed_texts = [self.text_processor.preprocess_for_embedding(text) for text in texts]
        
        # Transform using the pipeline
        embeddings = self.pipeline.transform(processed_texts)
        
        # Handle sparse vs dense matrices
        if hasattr(embeddings, 'toarray'):
            return embeddings.toarray().astype("float32")
        else:
            return embeddings.astype("float32")
    
    def get_semantic_similarity(self, text1, text2):
        """Get semantic similarity using the text processor."""
        return self.text_processor.calculate_semantic_similarity(text1, text2)