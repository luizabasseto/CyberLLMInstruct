#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberDataStructurer:
    def __init__(self, input_dir: str = "filtered_data", output_dir: str = "structured_data"):
        """Initialize the data structurer with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template patterns for different data types
        self.templates = {
            'vulnerability': {
                'instruction': [
                    "Explain the {cve_id} vulnerability and its potential impact.",
                    "What are the security implications of {cve_id}?",
                    "Describe the vulnerability identified as {cve_id}.",
                    "How does the {cve_id} vulnerability affect system security?"
                ],
                'response': "{description}\n\nImpact: {impact}\n\nAffected Systems: {affected_systems}"
            },
            'attack_pattern': {
                'instruction': [
                    "How does the {attack_name} attack work?",
                    "Explain the methodology of {attack_name}.",
                    "What is the {attack_name} attack pattern?",
                    "Describe the execution of {attack_name} attack."
                ],
                'response': "Attack Pattern: {description}\n\nTechniques Used: {techniques}\n\nMitigation: {mitigation}"
            },
            'security_advisory': {
                'instruction': [
                    "What are the recommended actions for {advisory_id}?",
                    "Explain the security advisory {advisory_id}.",
                    "What measures should be taken regarding {advisory_id}?",
                    "Describe the security implications and fixes for {advisory_id}."
                ],
                'response': "Advisory Details: {description}\n\nRecommended Actions: {recommendations}\n\nSeverity: {severity}"
            }
        }

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

    def detect_entry_type(self, entry: Dict) -> Optional[str]:
        """Detect the type of cybersecurity data entry."""
        # Check for CVE pattern
        if any(key.lower().startswith('cve-') for key in entry.keys()) or \
           any('cve-' in str(value).lower() for value in entry.values()):
            return 'vulnerability'
        
        # Check for attack pattern indicators
        if any(key in entry for key in {'attack_pattern', 'technique', 'tactic'}) or \
           any('attack' in str(value).lower() for value in entry.values()):
            return 'attack_pattern'
        
        # Check for security advisory
        if any(key in entry for key in {'advisory', 'bulletin', 'notice'}) or \
           any(isinstance(value, str) and 'advisory' in value.lower() for value in entry.values()):
            return 'security_advisory'
        
        return None

    def extract_fields(self, entry: Dict, entry_type: str) -> Dict:
        """Extract relevant fields based on entry type."""
        fields = {}
        
        if entry_type == 'vulnerability':
            # Extract CVE ID
            cve_pattern = r'CVE-\d{4}-\d{4,7}'
            cve_matches = []
            for value in entry.values():
                if isinstance(value, str):
                    matches = re.findall(cve_pattern, value, re.IGNORECASE)
                    cve_matches.extend(matches)
            
            fields['cve_id'] = cve_matches[0] if cve_matches else "Unknown CVE"
            
            # Extract other fields
            fields['description'] = self._find_field(entry, ['description', 'summary', 'details'])
            fields['impact'] = self._find_field(entry, ['impact', 'severity', 'criticality'])
            fields['affected_systems'] = self._find_field(entry, ['affected', 'affected_products', 'systems'])
            
        elif entry_type == 'attack_pattern':
            fields['attack_name'] = self._find_field(entry, ['name', 'title', 'pattern_name'])
            fields['description'] = self._find_field(entry, ['description', 'summary'])
            fields['techniques'] = self._find_field(entry, ['techniques', 'methods', 'tactics'])
            fields['mitigation'] = self._find_field(entry, ['mitigation', 'countermeasures', 'defense'])
            
        elif entry_type == 'security_advisory':
            fields['advisory_id'] = self._find_field(entry, ['id', 'advisory_id', 'bulletin_id'])
            fields['description'] = self._find_field(entry, ['description', 'summary', 'details'])
            fields['recommendations'] = self._find_field(entry, ['recommendations', 'solution', 'remediation'])
            fields['severity'] = self._find_field(entry, ['severity', 'criticality', 'risk_level'])
        
        return fields

    def _find_field(self, entry: Dict, possible_keys: List[str]) -> str:
        """Helper method to find a field value from possible keys."""
        for key in possible_keys:
            for entry_key, value in entry.items():
                if key.lower() in entry_key.lower() and isinstance(value, str):
                    return value
        return "Information not available"

    def create_instruction_response_pair(self, entry: Dict, entry_type: str) -> List[Dict]:
        """Create instruction-response pairs from an entry."""
        fields = self.extract_fields(entry, entry_type)
        template = self.templates[entry_type]
        pairs = []
        
        # Create multiple instruction-response pairs using different templates
        for instruction_template in template['instruction']:
            instruction = instruction_template.format(**fields)
            response = template['response'].format(**fields)
            
            pairs.append({
                'instruction': instruction,
                'response': response
            })
        
        return pairs

    def structure_dataset(self, input_file: Path) -> List[Dict]:
        """Structure the dataset into instruction-response format."""
        data = self.load_data(input_file)
        if not data:
            return []
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        structured_pairs = []
        for entry in data:
            entry_type = self.detect_entry_type(entry)
            if entry_type:
                pairs = self.create_instruction_response_pair(entry, entry_type)
                structured_pairs.extend(pairs)
        
        return structured_pairs

    def save_structured_data(self, data: List[Dict], original_file: Path):
        """Save structured data in both CSV and JSON formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{original_file.stem}_structured_{timestamp}"
        
        # Save as JSON
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved structured data to {json_path}")
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_name}.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        logger.info(f"Saved structured data to {csv_path}")

    def process_directory(self):
        """Process all files in the input directory."""
        total_pairs = 0
        processed_files = 0
        
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() not in {'.json', '.yaml', '.yml', '.csv'}:
                continue
            
            logger.info(f"Processing {file_path}")
            structured_pairs = self.structure_dataset(file_path)
            
            if structured_pairs:
                processed_files += 1
                total_pairs += len(structured_pairs)
                self.save_structured_data(structured_pairs, file_path)
        
        logger.info("Structuring Statistics:")
        logger.info(f"Processed Files: {processed_files}")
        logger.info(f"Total Instruction-Response Pairs: {total_pairs}")

def main():
    """Main function to demonstrate usage."""
    structurer = CyberDataStructurer()
    structurer.process_directory()

if __name__ == "__main__":
    main() 