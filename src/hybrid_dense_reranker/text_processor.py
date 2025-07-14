"""Advanced text preprocessing for improved RAG accuracy."""
import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass


class AdvancedTextProcessor:
    """Advanced text preprocessing for better semantic matching."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stop words
        self.stop_words.update(['said', 'say', 'says', 'telling', 'told', 'came', 'come', 'went', 'go'])
        
        # Legal/business terms that should be preserved
        self.preserve_terms = {
            'liability', 'contract', 'agreement', 'clause', 'indemnification',
            'revenue', 'financial', 'security', 'authentication', 'compliance',
            'risk', 'legal', 'obligation', 'breach', 'damages'
        }
        
        # Religious terms for Mormon corpus
        self.preserve_terms.update({
            'lord', 'god', 'nephi', 'faith', 'righteousness', 'commandment',
            'prayer', 'spirit', 'prophet', 'scripture', 'revelation'
        })
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but preserve important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        
        return text
    
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from text."""
        if not text:
            return set()
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Filter tokens
        keywords = set()
        for token in tokens:
            # Skip if too short, is stop word, or is punctuation
            if (len(token) < 3 or 
                token in self.stop_words or 
                token in string.punctuation):
                continue
            
            # Preserve important domain terms
            if token in self.preserve_terms:
                keywords.add(token)
            else:
                # Stem other words
                stemmed = self.stemmer.stem(token)
                if len(stemmed) >= 3:
                    keywords.add(stemmed)
        
        return keywords
    
    def preprocess_for_embedding(self, text: str) -> str:
        """Preprocess text specifically for TF-IDF embedding."""
        if not text:
            return ""
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Extract sentences and join important ones
        sentences = sent_tokenize(cleaned)
        
        # Filter out very short sentences
        meaningful_sentences = [s for s in sentences if len(s.split()) >= 3]
        
        if not meaningful_sentences:
            return cleaned
        
        # Join sentences back
        return ' '.join(meaningful_sentences)
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        if not query:
            return query
        
        # Define synonym mappings for common terms
        synonyms = {
            'risk': ['danger', 'threat', 'hazard', 'vulnerability'],
            'legal': ['law', 'juridical', 'regulatory', 'compliance'],
            'contract': ['agreement', 'deal', 'arrangement', 'compact'],
            'financial': ['monetary', 'fiscal', 'economic', 'budget'],
            'security': ['safety', 'protection', 'safeguard', 'defense'],
            'faith': ['belief', 'trust', 'confidence', 'devotion'],
            'righteousness': ['virtue', 'goodness', 'morality', 'piety'],
            'lord': ['god', 'deity', 'divine', 'almighty']
        }
        
        # Clean and tokenize query
        cleaned_query = self.clean_text(query)
        tokens = word_tokenize(cleaned_query)
        
        # Expand with synonyms
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            if token in synonyms:
                # Add one most relevant synonym
                expanded_tokens.append(synonyms[token][0])
        
        return ' '.join(expanded_tokens)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Extract keywords from both texts
        keywords1 = self.extract_keywords(text1)
        keywords2 = self.extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        if not union:
            return 0.0
        
        jaccard_sim = len(intersection) / len(union)
        
        # Boost score if important domain terms match
        important_matches = intersection.intersection(self.preserve_terms)
        if important_matches:
            boost = min(0.3, len(important_matches) * 0.1)
            jaccard_sim = min(1.0, jaccard_sim + boost)
        
        return jaccard_sim