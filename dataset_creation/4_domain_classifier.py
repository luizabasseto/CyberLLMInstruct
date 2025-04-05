#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDomainClassifier:
    def __init__(self, input_dir: str = "structured_data", output_dir: str = "domain_classified"):
        """Initialize the domain classifier with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define cybersecurity domains and their characteristics
        self.domains = {
            'malware': {
                'keywords': {
                    'malware', 'virus', 'trojan', 'ransomware', 'spyware', 'worm',
                    'backdoor', 'rootkit', 'keylogger', 'botnet', 'payload'
                },
                'patterns': [
                    r'malicious\s+(?:software|code|program)',
                    r'ransom(?:ware)?\s+attack',
                    r'malware\s+(?:infection|detection)'
                ]
            },
            'phishing': {
                'keywords': {
                    'phishing', 'spoofing', 'social engineering', 'impersonation',
                    'credential', 'theft', 'spam', 'scam', 'fraudulent'
                },
                'patterns': [
                    r'phishing\s+(?:email|campaign|attack)',
                    r'credential\s+(?:theft|stealing|harvesting)',
                    r'social\s+engineering'
                ]
            },
            'zero_day': {
                'keywords': {
                    'zero-day', '0day', 'unpatched', 'undisclosed', 'unknown vulnerability',
                    'novel attack', 'previously unknown'
                },
                'patterns': [
                    r'zero[\s-]day',
                    r'previously\s+unknown\s+vulnerability',
                    r'unpatched\s+(?:vulnerability|exploit)'
                ]
            },
            'iot_security': {
                'keywords': {
                    'iot', 'internet of things', 'smart device', 'embedded', 'firmware',
                    'sensor', 'connected device', 'smart home'
                },
                'patterns': [
                    r'iot\s+(?:device|security|vulnerability)',
                    r'internet\s+of\s+things',
                    r'smart\s+(?:device|home|sensor)'
                ]
            },
            'web_security': {
                'keywords': {
                    'xss', 'csrf', 'sql injection', 'web application', 'http',
                    'cookie', 'session', 'authentication', 'web server'
                },
                'patterns': [
                    r'cross[\s-]site\s+scripting',
                    r'sql\s+injection',
                    r'web\s+(?:application|server|security)'
                ]
            },
            'network_security': {
                'keywords': {
                    'ddos', 'firewall', 'packet', 'traffic', 'network', 'protocol',
                    'router', 'switch', 'gateway', 'ids', 'ips'
                },
                'patterns': [
                    r'denial[\s-]of[\s-]service',
                    r'network\s+(?:attack|security|protocol)',
                    r'traffic\s+(?:analysis|monitoring)'
                ]
            }
        }
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.model_path = self.output_dir / 'ml_model'
        self.model_path.mkdir(exist_ok=True)
        
        # Manual review tracking
        self.review_history = []

    def load_data(self, file_path: Path) -> Union[Dict, List, None]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif suffix in {'.yaml', '.yml'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif suffix == '.csv':
                return pd.read_csv(file_path).to_dict('records')
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def rule_based_classification(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify text using rule-based approach.
        Returns list of (domain, confidence) tuples.
        """
        text_lower = text.lower()
        scores = []
        
        for domain, rules in self.domains.items():
            score = 0
            # Check keywords
            for keyword in rules['keywords']:
                if keyword in text_lower:
                    score += 1
            
            # Check patterns
            for pattern in rules['patterns']:
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            
            # Normalize score
            max_possible = len(rules['keywords']) + len(rules['patterns'])
            confidence = score / max_possible if max_possible > 0 else 0
            
            if confidence > 0:
                scores.append((domain, confidence))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def train_ml_model(self, training_data: List[Dict]):
        """Train ML model on labeled data."""
        texts = []
        labels = []
        
        for entry in training_data:
            if 'text' in entry and 'domain' in entry:
                texts.append(entry['text'])
                labels.append(entry['domain'])
        
        if not texts:
            logger.warning("No training data available")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        X_test_vec = self.vectorizer.transform(X_test)
        accuracy = self.classifier.score(X_test_vec, y_test)
        logger.info(f"ML Model Accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(self.classifier, self.model_path / 'classifier.joblib')
        joblib.dump(self.vectorizer, self.model_path / 'vectorizer.joblib')

    def ml_classification(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify text using ML approach.
        Returns list of (domain, confidence) tuples.
        """
        try:
            # Load models if not initialized
            if not hasattr(self.classifier, 'classes_'):
                self.classifier = joblib.load(self.model_path / 'classifier.joblib')
                self.vectorizer = joblib.load(self.model_path / 'vectorizer.joblib')
            
            # Vectorize and predict
            text_vec = self.vectorizer.transform([text])
            probabilities = self.classifier.predict_proba(text_vec)[0]
            
            # Get predictions with confidence
            predictions = [
                (domain, prob) 
                for domain, prob in zip(self.classifier.classes_, probabilities)
                if prob > 0.1  # Filter low confidence predictions
            ]
            
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"ML classification error: {str(e)}")
            return []

    def combine_classifications(self, rule_based: List[Tuple[str, float]], 
                              ml_based: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine rule-based and ML classifications."""
        combined_scores = {}
        
        # Weight for each approach
        rule_weight = 0.6
        ml_weight = 0.4
        
        # Combine rule-based scores
        for domain, score in rule_based:
            combined_scores[domain] = score * rule_weight
        
        # Combine ML-based scores
        for domain, score in ml_based:
            if domain in combined_scores:
                combined_scores[domain] += score * ml_weight
            else:
                combined_scores[domain] = score * ml_weight
        
        # Sort by score
        return sorted(
            [(domain, score) for domain, score in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

    def classify_entry(self, entry: Dict) -> Dict:
        """Classify a single entry."""
        # Combine instruction and response for classification
        text = f"{entry.get('instruction', '')} {entry.get('response', '')}"
        
        # Get classifications
        rule_based = self.rule_based_classification(text)
        ml_based = self.ml_classification(text)
        
        # Combine results
        combined = self.combine_classifications(rule_based, ml_based)
        
        # Add classifications to entry
        entry['domains'] = [
            {'name': domain, 'confidence': float(confidence)}
            for domain, confidence in combined
            if confidence > 0.2  # Minimum confidence threshold
        ]
        
        # Add primary domain
        if entry['domains']:
            entry['primary_domain'] = entry['domains'][0]['name']
        else:
            entry['primary_domain'] = 'uncategorized'
        
        return entry

    def manual_review(self, entry: Dict) -> Dict:
        """Provide interface for manual review and correction."""
        print("\nEntry for Review:")
        print(f"Instruction: {entry.get('instruction', '')}")
        print(f"Response: {entry.get('response', '')}")
        print("\nAssigned Domains:")
        for domain in entry.get('domains', []):
            print(f"- {domain['name']} (confidence: {domain['confidence']:.2f})")
        
        correction = input("\nEnter correct domain(s) (comma-separated) or press Enter to accept: ")
        
        if correction.strip():
            domains = [d.strip() for d in correction.split(',')]
            entry['domains'] = [{'name': d, 'confidence': 1.0} for d in domains]
            entry['primary_domain'] = domains[0]
            entry['manually_reviewed'] = True
            
            # Record the correction for model improvement
            self.review_history.append({
                'text': f"{entry.get('instruction', '')} {entry.get('response', '')}",
                'original_classification': entry.get('primary_domain'),
                'corrected_classification': domains[0],
                'timestamp': datetime.now().isoformat()
            })
        
        return entry

    def save_classified_data(self, data: List[Dict], original_file: Path):
        """Save classified data in multiple formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{original_file.stem}_classified_{timestamp}"
        
        # Save complete data as JSON
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_name}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Save domain-specific files
        for domain in self.domains:
            domain_entries = [
                entry for entry in data 
                if entry.get('primary_domain') == domain
            ]
            if domain_entries:
                domain_path = self.output_dir / f"{base_name}_{domain}.json"
                with open(domain_path, 'w', encoding='utf-8') as f:
                    json.dump(domain_entries, f, indent=2)
        
        logger.info(f"Saved classified data to {self.output_dir}")

    def process_directory(self, enable_manual_review: bool = False):
        """Process all files in the input directory."""
        all_classified_data = []
        processed_files = 0
        
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() not in {'.json', '.yaml', '.yml', '.csv'}:
                continue
            
            logger.info(f"Processing {file_path}")
            data = self.load_data(file_path)
            
            if not data:
                continue
            
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            
            # Classify each entry
            classified_data = []
            for entry in data:
                classified_entry = self.classify_entry(entry)
                if enable_manual_review:
                    classified_entry = self.manual_review(classified_entry)
                classified_data.append(classified_entry)
            
            # Save results
            if classified_data:
                processed_files += 1
                all_classified_data.extend(classified_data)
                self.save_classified_data(classified_data, file_path)
        
        # Log statistics
        domain_counts = {}
        for entry in all_classified_data:
            domain = entry.get('primary_domain', 'uncategorized')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("\nClassification Statistics:")
        logger.info(f"Processed Files: {processed_files}")
        logger.info(f"Total Entries: {len(all_classified_data)}")
        logger.info("\nDomain Distribution:")
        for domain, count in sorted(domain_counts.items()):
            percentage = (count / len(all_classified_data)) * 100
            logger.info(f"{domain}: {count} entries ({percentage:.1f}%)")

def main():
    """Main function to demonstrate usage."""
    classifier = CyberDomainClassifier()
    
    # Enable manual review through command line argument
    import argparse
    parser = argparse.ArgumentParser(description='Classify cybersecurity data into domains')
    parser.add_argument('--manual-review', action='store_true', 
                       help='Enable manual review of classifications')
    args = parser.parse_args()
    
    classifier.process_directory(enable_manual_review=args.manual_review)

if __name__ == "__main__":
    main() 