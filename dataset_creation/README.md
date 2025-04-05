# Dataset Creation Pipeline

This directory contains a series of Python scripts that form a comprehensive pipeline for creating, processing, and validating a cyber security dataset for the CyberLLMInstruct paper ([arXiv:2503.09334](https://arxiv.org/abs/2503.09334)). Each script performs a specific function in the pipeline, designed to be run sequentially.

## Pipeline Overview

1. **Data Collection** (`1_data_collector.py`)
   - Collects raw data from various sources
   - Supports multiple input formats (JSON, CSV, YAML)
   - Handles API rate limiting and error recovery
   - Saves raw data with source metadata

2. **Data Filtering** (`2_data_filter.py`)
   - Removes irrelevant or low-quality entries
   - Applies content filters and quality checks
   - Handles duplicate detection
   - Generates filtering statistics

3. **Data Structuring** (`3_data_structurer.py`)
   - Standardises data format
   - Validates data structure
   - Normalises text content
   - Ensures consistent metadata fields

4. **Domain Classification** (`4_domain_classifier.py`)
   - Categorises entries into cyber security domains
   - Uses both rule-based and ML approaches
   - Supports manual classification corrections
   - Tracks classification confidence scores

5. **Manual Review** (`5_manual_reviewer.py`)
   - Interactive interface for data review
   - Quality assessment tools
   - Annotation capabilities
   - Progress tracking and reporting

6. **Security Alignment** (`6_security_aligner.py`)
   - Adds security-focused examples
   - Generates adversarial cases
   - Implements compliance testing
   - Flags sensitive content

7. **AI Enhancement** (`7_ai_tuner.py`)
   - Improves entry quality using AI
   - Enhances clarity and coherence
   - Handles API integration
   - Tracks enhancement history

8. **Final Assembly** (`8_final_assembler.py`)
   - Merges processed data
   - Performs final validation
   - Removes duplicates
   - Generates comprehensive reports
   - Exports in multiple formats

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run scripts in sequence:
```bash
# 1. Collect data
python 1_data_collector.py --sources source1 source2 --output-dir raw_data

# 2. Filter data
python 2_data_filter.py --input-dir raw_data --output-dir filtered_data

# 3. Structure data
python 3_data_structurer.py --input-dir filtered_data --output-dir structured_data

# 4. Classify domains
python 4_domain_classifier.py --input-dir structured_data --output-dir classified_data

# 5. Manual review
python 5_manual_reviewer.py --input-dir classified_data --output-dir reviewed_data

# 6. Add security examples
python 6_security_aligner.py --input-dir reviewed_data --output-dir security_aligned

# 7. Enhance with AI
python 7_ai_tuner.py --input-dir security_aligned --output-dir ai_enhanced

# 8. Final assembly
python 8_final_assembler.py --input-dirs ai_enhanced reviewed_data --output-dir final_dataset
```

## Output Formats

The final dataset is available in the following formats:
- JSON (with full metadata)
- Parquet (efficient storage)

Both formats contain complete metadata and are fully compatible with the dataset processing pipeline.

## Data Format

Each entry in the dataset follows this structure:
```json
{
  "instruction": "text of the instruction",
  "response": "text of the response",
  "metadata": {
    "category": "cybersecurity_domain",
    "subcategory": "specific_topic",
    "security_flags": {
      "review_required": boolean,
      "isolation_required": boolean
    },
    "source": "origin_of_data",
    "processing_history": [
      {
        "step": "processing_step",
        "timestamp": "ISO-8601 timestamp"
      }
    ]
  }
}