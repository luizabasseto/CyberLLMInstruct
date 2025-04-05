#!/usr/bin/env python3

import json
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Set, Tuple
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDataFilter:
    def __init__(self, input_dir: str = "raw_data", output_dir: str = "filtered_data"):
        """Initialize the data filter with input and output directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Keywords and patterns for filtering
        self.cybersecurity_keywords = {
            'high_relevance': {
                'vulnerability', 'exploit', 'malware', 'ransomware', 'cyber', 'security',
                'attack', 'threat', 'breach', 'CVE-', 'patch', 'authentication', 'authorization',
                'encryption', 'cryptography', 'backdoor', 'botnet', 'phishing', 'injection',
                'zero-day', '0day', 'penetration', 'pentest', 'firewall', 'malicious'
            },
            'medium_relevance': {
                'network', 'system', 'software', 'hardware', 'protocol', 'server',
                'client', 'database', 'web', 'application', 'code', 'programming',
                'access', 'control', 'monitoring', 'detection', 'response', 'incident'
            }
        }
        
        # Patterns to identify low-quality or irrelevant entries
        self.exclusion_patterns = {
            'generic_terms': r'\b(test|sample|example|dummy|todo)\b',
            'placeholder_text': r'\b(lorem ipsum|xxx|placeholder)\b',
            'empty_content': r'^\s*$',
        }
        
        # Minimum content requirements
        self.min_content_length = 50  # characters
        self.min_keyword_matches = 2

    def load_data(self, file_path: Path) -> Union[Dict, List, None]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif suffix == '.yaml' or suffix == '.yml':
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

    def is_relevant_content(self, text: str) -> bool:
        """
        Check if the content is relevant to cybersecurity.
        Returns True if content is relevant, False otherwise.
        """
        if not isinstance(text, str):
            return False
            
        # Check minimum length
        if len(text) < self.min_content_length:
            return False
            
        # Check for exclusion patterns
        for pattern in self.exclusion_patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Count keyword matches
        text_lower = text.lower()
        high_relevance_matches = sum(1 for kw in self.cybersecurity_keywords['high_relevance'] 
                                   if kw.lower() in text_lower)
        medium_relevance_matches = sum(1 for kw in self.cybersecurity_keywords['medium_relevance'] 
                                     if kw.lower() in text_lower)
        
        # Scoring system: high relevance keywords count as 2, medium as 1
        relevance_score = (high_relevance_matches * 2) + medium_relevance_matches
        
        return relevance_score >= self.min_keyword_matches

    def filter_entry(self, entry: Dict) -> Tuple[bool, str]:
        """
        Filter a single entry based on relevance criteria.
        Returns (is_relevant, reason).
        """
        # Combine all text fields for analysis
        text_content = []
        for key, value in entry.items():
            if isinstance(value, str):
                text_content.append(value)
            elif isinstance(value, (dict, list)):
                text_content.append(str(value))
        
        combined_text = ' '.join(text_content)
        
        if not combined_text.strip():
            return False, "Empty content"
        
        if not self.is_relevant_content(combined_text):
            return False, "Not relevant to cybersecurity"
            
        return True, "Relevant"

    def filter_dataset(self, input_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter the dataset and return (relevant_entries, filtered_out_entries).
        """
        data = self.load_data(input_file)
        if not data:
            return [], []
        
        # Convert to list if dictionary
        if isinstance(data, dict):
            data = [data]
        
        relevant_entries = []
        filtered_out_entries = []
        
        for entry in data:
            is_relevant, reason = self.filter_entry(entry)
            if is_relevant:
                relevant_entries.append(entry)
            else:
                entry['filtered_reason'] = reason
                filtered_out_entries.append(entry)
        
        return relevant_entries, filtered_out_entries

    def save_filtered_data(self, data: List[Dict], original_file: Path, suffix: str = '') -> bool:
        """Save filtered data with original format."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"{original_file.stem}{suffix}_{timestamp}{original_file.suffix}"
            
            if original_file.suffix.lower() == '.json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif original_file.suffix.lower() in {'.yaml', '.yml'}:
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True)
            elif original_file.suffix.lower() == '.csv':
                pd.DataFrame(data).to_csv(output_file, index=False)
            
            logger.info(f"Saved filtered data to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving filtered data: {str(e)}")
            return False

    def process_directory(self):
        """Process all files in the input directory."""
        total_stats = {
            'processed_files': 0,
            'total_entries': 0,
            'retained_entries': 0,
            'filtered_entries': 0
        }
        
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() not in {'.json', '.yaml', '.yml', '.csv'}:
                continue
                
            logger.info(f"Processing {file_path}")
            relevant_entries, filtered_entries = self.filter_dataset(file_path)
            
            if relevant_entries or filtered_entries:
                total_stats['processed_files'] += 1
                total_stats['total_entries'] += len(relevant_entries) + len(filtered_entries)
                total_stats['retained_entries'] += len(relevant_entries)
                total_stats['filtered_entries'] += len(filtered_entries)
                
                # Save relevant entries
                if relevant_entries:
                    self.save_filtered_data(relevant_entries, file_path, '_filtered')
                
                # Optionally save filtered-out entries for analysis
                if filtered_entries:
                    self.save_filtered_data(filtered_entries, file_path, '_removed')
        
        # Log statistics
        logger.info("Filtering Statistics:")
        logger.info(f"Processed Files: {total_stats['processed_files']}")
        logger.info(f"Total Entries: {total_stats['total_entries']}")
        logger.info(f"Retained Entries: {total_stats['retained_entries']}")
        logger.info(f"Filtered Entries: {total_stats['filtered_entries']}")
        logger.info(f"Retention Rate: {(total_stats['retained_entries'] / total_stats['total_entries'] * 100):.2f}%")

def main():
    """Main function to demonstrate usage."""
    filter = CyberDataFilter()
    filter.process_directory()

if __name__ == "__main__":
    main() 