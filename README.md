# CyberLLMInstruct

Code and dataset repository for the paper:
> CyberLLMInstruct: A new dataset for analysing safety of fine-tuned LLMs using cyber security data

Published at: [arXiv:2503.09334](https://arxiv.org/abs/2503.09334)

This repository contains all code and materials to reproduce the dataset used in the paper. Due to copyright considerations, we provide scripts to regenerate the dataset rather than distributing it directly.

## Repository Structure

* `dataset_creation/`: Dataset creation pipeline
  - Eight sequential scripts (`1_data_collector.py` through `8_final_assembler.py`) for collecting, processing, and validating cyber security data
  - See [`dataset_creation/README.md`](dataset_creation/README.md) for detailed pipeline documentation
  - Use these scripts to reproduce the dataset following our methodology

* `examples/`: Examples of using the CyberLLMInstruct dataset
  - `deepeval/`: Example 1
  - `cybermetric/`: Example 2

* `finetune/`: Comprehensive fine-tuning pipeline
  - `data_prep.py`: Data preprocessing for various LLM architectures
  - `train.py`: Training script with support for LoRA and quantisation
  - `inference.py`: Inference script with interactive and batch modes
  - `checkpoint_manager.py`: Checkpoint management utilities
  - See [`finetune/README.md`](finetune/README.md) for detailed fine-tuning documentation
  
* `scripts/`: Utility scripts for dataset management
  - `categorise.py`: Pattern-based domain categorisation
  - `dataset_export.py`: Dataset export and platform upload
  - See [`scripts/README.md`](scripts/README.md) for usage instructions

## Supported Models

The following large language models have been fine-tuned on the CyberLLMInstruct dataset:
- Phi 3 Mini 3.8B
- Mistral 7B
- Qwen 2.5 7B
- Llama 3 8B
- Llama 3.1 8B
- Gemma 2 9B
- Llama 2 70B

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{elzemity2025cyberllminstructnewdatasetanalysing,
      title={CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data}, 
      author={Adel ElZemity and Budi Arief and Shujun Li},
      year={2025},
      eprint={2503.09334},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2503.09334}, 
}
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/adelsamir01/CyberLLMInstruct.git
cd CyberLLMInstruct
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Dataset Creation:
   To create the dataset, follow the pipeline in the `dataset_creation/` directory:
   - Each script (1-8) should be run sequentially
   - Detailed instructions are provided in `dataset_creation/README.md`
   - This process ensures compliance with data usage rights and allows you to reproduce the dataset

4. Follow the specific documentation in each directory for:
   - Fine-tuning models: See `finetune/README.md`
   - Model evaluation: See `evaluation/README.md`
   - Utility scripts: See `scripts/README.md`
