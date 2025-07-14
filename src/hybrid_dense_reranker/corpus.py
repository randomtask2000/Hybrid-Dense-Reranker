"""Enhanced corpus loading and management with smart chunking."""
import os
import re
from typing import List, Dict
from .config import CORPUS_SOURCE, CHUNK_SIZE, CHUNK_OVERLAP
from .text_processor import AdvancedTextProcessor


def load_mormon_corpus():
    """Load and chunk the Mormon text from the data file."""
    try:
        # Look for data file in project root
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'mormon13short.txt')
        with open(data_path, 'r', encoding='utf-8') as file:
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
                    "content": current_chunk.strip(),
                    "chunk_id": chunk_id,
                    "source": "mormon"
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
                "content": current_chunk.strip(),
                "chunk_id": chunk_id,
                "source": "mormon"
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
    """Return an enhanced default sample corpus with more comprehensive content."""
    return [
        {
            "title": "Legal Risk Assessment - Contract Liability", 
            "content": "The current service agreement exposes our organization to significant liability risks due to several deficiencies. First, the contract lacks proper indemnification clauses that would protect us from third-party claims. Second, there are insufficient limitations on consequential damages, potentially exposing us to unlimited liability. The agreement also fails to include adequate force majeure provisions and dispute resolution mechanisms. Legal counsel recommends immediate contract revision to include comprehensive liability protection, capped damages clauses, and clear termination procedures. Without these protections, the organization faces substantial financial and legal exposure.",
            "chunk_id": 1,
            "source": "default"
        },
        {
            "title": "Cybersecurity Policy - Authentication Requirements", 
            "content": "All employees must implement multi-factor authentication (2FA) to reduce unauthorized access risks to corporate systems. This security requirement applies to email accounts, cloud storage platforms, financial systems, and any applications containing sensitive data. The IT department has identified that single-factor authentication presents significant vulnerabilities, particularly given the increase in phishing attacks and credential theft. Implementation of 2FA has been shown to prevent 99.9% of automated cyber attacks. Failure to comply with authentication requirements may result in disciplinary action and potential security incidents that could compromise client data and corporate intellectual property.",
            "chunk_id": 2,
            "source": "default"
        },
        {
            "title": "Financial Performance Analysis - Q3 2023", 
            "content": "Revenue performance for Q3 2023 shows strong growth of 15% year-over-year, driven primarily by increased client acquisitions and expanded service offerings. However, operational expenses have also increased significantly, particularly in legal and compliance costs due to ongoing litigation matters. Legal expenses alone account for 8% of total operational costs this quarter, representing a 45% increase from the previous year. The litigation involves contract disputes, regulatory compliance issues, and intellectual property claims. While revenue growth remains positive, the company must address escalating legal costs through improved contract management, enhanced compliance procedures, and proactive risk mitigation strategies to maintain profitability.",
            "chunk_id": 3,
            "source": "default"
        },
        {
            "title": "Risk Management Framework - Enterprise Security", 
            "content": "The enterprise risk management framework identifies several critical areas requiring immediate attention: cybersecurity threats, regulatory compliance gaps, and operational vulnerabilities. Cybersecurity risks include data breaches, ransomware attacks, and insider threats. The security assessment reveals insufficient monitoring capabilities, outdated encryption protocols, and inadequate incident response procedures. Regulatory compliance risks span multiple jurisdictions with varying data protection requirements, financial reporting standards, and industry-specific regulations. Operational risks include supply chain disruptions, key personnel dependencies, and technology infrastructure failures. Mitigation strategies must address these interconnected risks through comprehensive policies, regular audits, and continuous monitoring systems.",
            "chunk_id": 4,
            "source": "default"
        },
        {
            "title": "Compliance Audit Results - Data Protection", 
            "content": "The annual compliance audit reveals several areas of non-compliance with data protection regulations including GDPR, CCPA, and industry-specific privacy requirements. Key findings include inadequate data encryption for customer information, insufficient access controls for sensitive databases, and incomplete data breach notification procedures. The audit also identified gaps in employee training regarding data handling practices and inadequate documentation of data processing activities. Immediate corrective actions required include implementation of enhanced encryption protocols, revision of privacy policies, comprehensive staff training programs, and establishment of a dedicated data protection officer role. Failure to address these compliance gaps could result in significant regulatory penalties and reputational damage.",
            "chunk_id": 5,
            "source": "default"
        }
    ]


def load_corpus():
    """Load the corpus based on configuration."""
    if CORPUS_SOURCE.lower() == "mormon":
        return load_mormon_corpus()
    else:
        return get_default_corpus()