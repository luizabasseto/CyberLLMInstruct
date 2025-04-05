#!/usr/bin/env python3

import json
import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import time
import asyncio
import aiohttp
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import difflib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AITuner:
    def __init__(self, input_dir: str = "security_aligned", output_dir: str = "ai_enhanced",
                 api_key: Optional[str] = None, model: str = "gpt-4-1106-preview"):
        """Initialize the AI tuner with directory and API configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Initialize rich console
        self.console = Console()
        
        # Enhancement prompts
        self.enhancement_prompts = {
            'clarity': '''
Analyze and enhance the following cybersecurity instruction-response pair for clarity and precision.
Focus on making technical concepts clear and actionable while maintaining security context.

Original Instruction: {instruction}
Original Response: {response}

Please provide enhanced versions that:
1. Use clear, precise language
2. Maintain technical accuracy
3. Include specific details and examples
4. Follow cybersecurity best practices
5. Keep the security context intact

Return the enhanced versions in JSON format:
{
    "enhanced_instruction": "...",
    "enhanced_response": "...",
    "changes_made": ["list of specific changes"],
    "enhancement_notes": "explanation of improvements"
}
''',
            'coherence': '''
Review and improve the coherence and logical flow of this cybersecurity instruction-response pair.
Ensure the response directly addresses the instruction and follows a logical structure.

Original Instruction: {instruction}
Original Response: {response}

Provide enhanced versions that:
1. Ensure logical flow and progression
2. Strengthen the connection between instruction and response
3. Add transitional elements where needed
4. Maintain consistent terminology
5. Preserve security context and accuracy

Return the enhanced versions in JSON format:
{
    "enhanced_instruction": "...",
    "enhanced_response": "...",
    "changes_made": ["list of specific changes"],
    "enhancement_notes": "explanation of improvements"
}
''',
            'detail': '''
Enhance the level of detail and comprehensiveness in this cybersecurity instruction-response pair.
Add relevant technical details while maintaining clarity and practicality.

Original Instruction: {instruction}
Original Response: {response}

Provide enhanced versions that:
1. Add relevant technical details
2. Include specific examples or scenarios
3. Explain key concepts thoroughly
4. Reference relevant security standards or practices
5. Maintain balance between detail and clarity

Return the enhanced versions in JSON format:
{
    "enhanced_instruction": "...",
    "enhanced_response": "...",
    "changes_made": ["list of specific changes"],
    "enhancement_notes": "explanation of improvements"
}
'''
        }
        
        # Enhancement tracking
        self.enhancement_stats = {
            'processed_entries': 0,
            'enhanced_entries': 0,
            'failed_enhancements': 0,
            'total_tokens_used': 0,
            'enhancement_types': {
                'clarity': 0,
                'coherence': 0,
                'detail': 0
            }
        }

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def enhance_entry_with_ai(self, entry: Dict, enhancement_type: str) -> Optional[Dict]:
        """Enhance a single entry using AI with retry logic."""
        try:
            # Prepare the prompt
            prompt = self.enhancement_prompts[enhancement_type].format(
                instruction=entry.get('instruction', ''),
                response=entry.get('response', '')
            )
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert focused on improving dataset quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Update token usage
            self.enhancement_stats['total_tokens_used'] += response.usage.total_tokens
            
            # Parse the response
            try:
                enhanced_data = json.loads(response.choices[0].message.content)
                return enhanced_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing AI response: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in AI enhancement: {str(e)}")
            return None

    def calculate_text_diff(self, original: str, enhanced: str) -> List[str]:
        """Calculate and format the differences between original and enhanced text."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            enhanced.splitlines(keepends=True),
            fromfile='original',
            tofile='enhanced',
            n=0
        )
        return [line for line in diff if line.startswith(('+', '-'))]

    def create_enhancement_metadata(self, original: Dict, enhanced_data: Dict,
                                  enhancement_type: str) -> Dict:
        """Create metadata about the enhancements made."""
        instruction_diff = self.calculate_text_diff(
            original.get('instruction', ''),
            enhanced_data['enhanced_instruction']
        )
        response_diff = self.calculate_text_diff(
            original.get('response', ''),
            enhanced_data['enhanced_response']
        )
        
        return {
            'enhancement_type': enhancement_type,
            'timestamp': datetime.now().isoformat(),
            'changes_made': enhanced_data.get('changes_made', []),
            'enhancement_notes': enhanced_data.get('enhancement_notes', ''),
            'diffs': {
                'instruction': instruction_diff,
                'response': response_diff
            },
            'original_content': {
                'instruction': original.get('instruction', ''),
                'response': original.get('response', '')
            }
        }

    async def enhance_entry(self, entry: Dict) -> Dict:
        """Enhance a single entry with all enhancement types."""
        enhanced_entry = entry.copy()
        enhanced_entry['enhancement_history'] = []
        
        for enhancement_type in self.enhancement_prompts.keys():
            enhanced_data = await self.enhance_entry_with_ai(enhanced_entry, enhancement_type)
            
            if enhanced_data:
                # Update entry with enhanced content
                enhanced_entry['instruction'] = enhanced_data['enhanced_instruction']
                enhanced_entry['response'] = enhanced_data['enhanced_response']
                
                # Create and store enhancement metadata
                metadata = self.create_enhancement_metadata(
                    entry, enhanced_data, enhancement_type
                )
                enhanced_entry['enhancement_history'].append(metadata)
                
                # Update statistics
                self.enhancement_stats['enhancement_types'][enhancement_type] += 1
                
            time.sleep(1)  # Rate limiting
        
        return enhanced_entry

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

    def save_data(self, data: List[Dict], original_file: Path):
        """Save enhanced data with detailed metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{original_file.stem}_enhanced_{timestamp}"
        
        # Save complete data as JSON
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save a simplified CSV version (without enhancement history)
        csv_data = []
        for entry in data:
            csv_entry = {
                'instruction': entry['instruction'],
                'response': entry['response']
            }
            if 'metadata' in entry:
                csv_entry.update(entry['metadata'])
            csv_data.append(csv_entry)
        
        csv_path = self.output_dir / f"{base_name}.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        # Save enhancement statistics
        stats_path = self.output_dir / f"{base_name}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.enhancement_stats, f, indent=2)
        
        logger.info(f"Saved enhanced data to {self.output_dir}")

    async def process_file(self, file_path: Path):
        """Process a single file."""
        data = self.load_data(file_path)
        if not data:
            return
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        enhanced_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task(f"Enhancing entries...", total=len(data))
            
            for entry in data:
                self.enhancement_stats['processed_entries'] += 1
                
                try:
                    enhanced_entry = await self.enhance_entry(entry)
                    enhanced_data.append(enhanced_entry)
                    self.enhancement_stats['enhanced_entries'] += 1
                except Exception as e:
                    logger.error(f"Error enhancing entry: {str(e)}")
                    enhanced_data.append(entry)  # Keep original if enhancement fails
                    self.enhancement_stats['failed_enhancements'] += 1
                
                progress.update(task, advance=1)
        
        # Save enhanced data
        self.save_data(enhanced_data, file_path)

    async def process_directory(self):
        """Process all files in the input directory."""
        for file_path in self.input_dir.glob('*.*'):
            if file_path.suffix.lower() not in {'.json', '.yaml', '.yml', '.csv'}:
                continue
            
            logger.info(f"Processing {file_path}")
            await self.process_file(file_path)
        
        # Log final statistics
        logger.info("\nEnhancement Statistics:")
        logger.info(f"Processed Entries: {self.enhancement_stats['processed_entries']}")
        logger.info(f"Enhanced Entries: {self.enhancement_stats['enhanced_entries']}")
        logger.info(f"Failed Enhancements: {self.enhancement_stats['failed_enhancements']}")
        logger.info(f"Total Tokens Used: {self.enhancement_stats['total_tokens_used']}")
        logger.info("\nEnhancements by Type:")
        for enhancement_type, count in self.enhancement_stats['enhancement_types'].items():
            logger.info(f"{enhancement_type}: {count}")

async def main():
    """Main function to demonstrate usage."""
    import os
    
    # Get OpenAI API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Enhance dataset quality using AI')
    parser.add_argument('--model', type=str, default="gpt-4-1106-preview",
                       help='OpenAI model to use (default: gpt-4-1106-preview)')
    args = parser.parse_args()
    
    # Initialize and run the tuner
    tuner = AITuner(api_key=api_key, model=args.model)
    await tuner.process_directory()

if __name__ == "__main__":
    asyncio.run(main()) 