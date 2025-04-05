#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Set, Optional, Union
from datetime import datetime
import hashlib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import jsonschema
from concurrent.futures import ThreadPoolExecutor
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetAssembler:
    def __init__(self, input_dirs: List[str], output_dir: str = "final_dataset"):
        """Initialize the dataset assembler with directory configurations."""
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize rich console
        self.console = Console()
        
        # Dataset schema for validation
        self.dataset_schema = {
            "type": "object",
            "required": ["instruction", "response"],
            "properties": {
                "instruction": {"type": "string", "minLength": 1},
                "response": {"type": "string", "minLength": 1},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "subcategory": {"type": "string"},
                        "security_flags": {
                            "type": "object",
                            "properties": {
                                "review_required": {"type": "boolean"},
                                "isolation_required": {"type": "boolean"}
                            }
                        },
                        "enhancement_history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "enhancement_type": {"type": "string"},
                                    "timestamp": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Assembly statistics
        self.stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'duplicate_entries': 0,
            'invalid_entries': 0,
            'sources': set(),
            'categories': set(),
            'entry_lengths': {
                'instruction': [],
                'response': []
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
            elif suffix == '.parquet':
                return pd.read_parquet(file_path).to_dict('records')
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def validate_entry(self, entry: Dict) -> bool:
        """Validate a single entry against the schema."""
        try:
            jsonschema.validate(instance=entry, schema=self.dataset_schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False

    def generate_entry_hash(self, entry: Dict) -> str:
        """Generate a unique hash for an entry based on its content."""
        # Create a normalized string representation of instruction and response
        content = (
            entry.get('instruction', '').lower().strip() +
            entry.get('response', '').lower().strip()
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def process_entry(self, entry: Dict, source_file: Path) -> Optional[Dict]:
        """Process and validate a single entry."""
        try:
            # Clean text fields
            entry['instruction'] = self.clean_text(entry.get('instruction', ''))
            entry['response'] = self.clean_text(entry.get('response', ''))
            
            # Validate entry
            if not self.validate_entry(entry):
                self.stats['invalid_entries'] += 1
                return None
            
            # Add source information if not present
            if 'metadata' not in entry:
                entry['metadata'] = {}
            entry['metadata']['source_file'] = str(source_file)
            entry['metadata']['processing_timestamp'] = datetime.now().isoformat()
            
            # Update statistics
            if 'category' in entry.get('metadata', {}):
                self.stats['categories'].add(entry['metadata']['category'])
            
            self.stats['entry_lengths']['instruction'].append(len(entry['instruction']))
            self.stats['entry_lengths']['response'].append(len(entry['response']))
            
            return entry
            
        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            self.stats['invalid_entries'] += 1
            return None

    def remove_duplicates(self, entries: List[Dict]) -> List[Dict]:
        """Remove duplicate entries based on content hash."""
        unique_entries = {}
        duplicates = 0
        
        for entry in entries:
            entry_hash = self.generate_entry_hash(entry)
            if entry_hash not in unique_entries:
                unique_entries[entry_hash] = entry
            else:
                duplicates += 1
        
        self.stats['duplicate_entries'] = duplicates
        return list(unique_entries.values())

    def save_dataset(self, data: List[Dict], format: str):
        """Save dataset in specified format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"cybersecurity_dataset_{timestamp}"
        
        if format == 'json':
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved JSON dataset to {json_path}")
            
        elif format == 'csv':
            csv_path = self.output_dir / f"{base_name}.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV dataset to {csv_path}")
            
        elif format == 'parquet':
            parquet_path = self.output_dir / f"{base_name}.parquet"
            table = pa.Table.from_pandas(pd.DataFrame(data))
            pq.write_table(table, parquet_path)
            logger.info(f"Saved Parquet dataset to {parquet_path}")

    def generate_dataset_report(self):
        """Generate a detailed report about the assembled dataset."""
        report = Table(title="Dataset Assembly Report", show_header=True)
        report.add_column("Metric", style="cyan")
        report.add_column("Value", style="green")
        
        # Add statistics
        report.add_row("Total Entries Processed", str(self.stats['total_entries']))
        report.add_row("Valid Entries", str(self.stats['valid_entries']))
        report.add_row("Duplicate Entries", str(self.stats['duplicate_entries']))
        report.add_row("Invalid Entries", str(self.stats['invalid_entries']))
        report.add_row("Number of Sources", str(len(self.stats['sources'])))
        report.add_row("Number of Categories", str(len(self.stats['categories'])))
        
        # Calculate averages
        avg_instruction_len = sum(self.stats['entry_lengths']['instruction']) / len(self.stats['entry_lengths']['instruction'])
        avg_response_len = sum(self.stats['entry_lengths']['response']) / len(self.stats['entry_lengths']['response'])
        
        report.add_row("Average Instruction Length", f"{avg_instruction_len:.1f} characters")
        report.add_row("Average Response Length", f"{avg_response_len:.1f} characters")
        
        self.console.print(report)
        
        # Save report
        report_path = self.output_dir / f"dataset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(str(report))

    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file."""
        data = self.load_data(file_path)
        if not data:
            return []
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        self.stats['total_entries'] += len(data)
        self.stats['sources'].add(str(file_path))
        
        # Process entries
        processed_entries = []
        for entry in data:
            processed_entry = self.process_entry(entry, file_path)
            if processed_entry:
                processed_entries.append(processed_entry)
                self.stats['valid_entries'] += 1
        
        return processed_entries

    def assemble_dataset(self):
        """Assemble the final dataset from all sources."""
        all_entries = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            # Process all files in parallel
            with ThreadPoolExecutor() as executor:
                file_paths = list(itertools.chain.from_iterable(
                    d.glob('*.*') for d in self.input_dirs
                ))
                
                task = progress.add_task("Processing files...", total=len(file_paths))
                
                for entries in executor.map(self.process_file, file_paths):
                    all_entries.extend(entries)
                    progress.update(task, advance=1)
        
        # Remove duplicates
        unique_entries = self.remove_duplicates(all_entries)
        
        # Save in multiple formats
        for format in ['json', 'csv', 'parquet']:
            self.save_dataset(unique_entries, format)
        
        # Generate report
        self.generate_dataset_report()
        
        return unique_entries

def main():
    """Main function to demonstrate usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Assemble final cybersecurity dataset')
    parser.add_argument('--input-dirs', nargs='+', required=True,
                       help='Input directories containing processed data')
    parser.add_argument('--output-dir', default='final_dataset',
                       help='Output directory for final dataset')
    args = parser.parse_args()
    
    assembler = DatasetAssembler(args.input_dirs, args.output_dir)
    assembler.assemble_dataset()

if __name__ == "__main__":
    main() 