# Automated Medical Impression Generation

A fine-tuned approach for generating radiology impressions from clinical findings using Microsoft's MediPhi-Instruct model with LoRA adapters.

## 🎯 Project Overview

This project demonstrates an end-to-end approach to automated medical impression generation that achieves significant performance improvements over baseline models. The solution focuses on findings-rich content filtering and quality-based segmentation to produce clinically relevant impressions.

### Key Results
- **19.6% improvement** in ROUGE-1 scores over base model
- **56.6% improvement** in ROUGE-2 scores
- **35.1% improvement** for MR imaging (largest modality)
- Systematic evaluation across **7 imaging modalities**

## 📁 Project Structure

```
├── report/                          # Technical assessment report
│   └── report.tex                   # LaTeX source for final report
├── notebooks/                       # Jupyter notebooks
│   ├── 00_eda.ipynb                # Exploratory data analysis
│   ├── 03_impression_model_ft.ipynb # Model fine-tuning
│   └── 04_model_evaluation.ipynb   # Model evaluation
├── data_preprocessing/              # Data processing scripts
│   ├── enhanced_preprocessing.py   # Findings-focused preprocessing
│   └── analyze_filtering.py       # Data quality analysis
├── evaluation/                     # Evaluation frameworks
│   ├── systematic_evaluation.py   # Systematic modality evaluation
│   ├── comparative_evaluator.py   # Base vs fine-tuned comparison
│   └── quick_modality_eval.py     # Simplified evaluation
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- HuggingFace account for model access

### Installation

```bash
# Clone repository
git clone https://github.com/msahamed/fine_tuning_radiology_impression_generation
cd sirona_medical

# Install dependencies
pip install torch transformers datasets evaluate peft accelerate bitsandbytes
pip install pandas numpy scikit-learn rouge-score
```

### Data Preparation

**Load your radiology data** in CSV format with columns:
   - `structured_report_dict`: Parsed clinical content
   - `clinic_id`: Institution identifier
   - `modality`: Imaging type (MR, CT, CR, etc.)

### Model Fine-Tuning

Key configuration:
- **Base Model**: `microsoft/MediPhi-Instruct`
- **Method**: LoRA with r=8, alpha=32
- **Training**: 1 epoch, 2e-4 learning rate
- **Hardware**: Single RTX 4090 GPU


## 📊 Results Summary

### Overall Performance
| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| ROUGE-1 | 0.3465 | 0.4146 | +19.6% |
| ROUGE-2 | 0.1800 | 0.2818 | +56.6% |
| ROUGE-L | 0.2727 | 0.3720 | +36.4% |

### Performance by Modality
| Modality | Base ROUGE-1 | Fine-Tuned ROUGE-1 | Improvement |
|----------|---------------|---------------------|-------------|
| MR | 0.4642 | 0.6274 | +35.1% |
| CR | 0.3283 | 0.3970 | +20.9% |
| XR | 0.2859 | 0.3812 | +33.3% |
| CT | 0.2836 | 0.2978 | +5.0% |
| US | 0.3073 | 0.3394 | +10.4% |

## 🔬 Technical Approach

### Data Processing Pipeline
1. **Quality Filtering**: Focus on findings-rich reports (≥100 characters)
2. **Essential Field Extraction**: Findings and impressions only
3. **Modality Grouping**: Consolidate rare imaging types
4. **Stratified Splitting**: Maintain clinic-modality distributions

### Model Architecture
- **Base Model**: Microsoft MediPhi-Instruct (3B parameters)
- **Fine-Tuning**: LoRA adapters (0.33% trainable parameters)
- **Quantization**: 4-bit NF4 for memory efficiency
- **Training Data**: 8,865 findings-rich samples

### Evaluation Framework
- **Systematic Sampling**: 20 samples per modality
- **Comparative Analysis**: Base vs fine-tuned models
- **Metrics**: ROUGE-1/2/L scores
- **Reproducibility**: Fixed random seeds

## 📈 Key Insights

1. **Modality-Specific Gains**: High-volume modalities (MR, CR, XR) show strongest improvements
2. **Data Quality Impact**: Findings-rich filtering crucial for performance
3. **Efficient Training**: LoRA enables effective fine-tuning with minimal parameters
4. **Clinical Relevance**: Improvements concentrated in major imaging types (>85% of cases)

## 🔗 Model Access

The fine-tuned adapter is available on Hugging Face Hub:
**[sabber/medphi-radiology-summary-adapter](https://huggingface.co/sabber/medphi-radiology-summary-adapter)**

### Usage Example
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Load fine-tuned model
model = AutoPeftModelForCausalLM.from_pretrained(
    "sabber/medphi-radiology-summary-adapter",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("sabber/medphi-radiology-summary-adapter")

# Generate impression
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
messages = [
    {"role": "system", "content": "You are an expert radiologist..."},
    {"role": "user", "content": "[CLINIC: clinic_1] [MODALITY: MR] FINDINGS: ..."}
]
output = pipe(messages, max_new_tokens=300, temperature=0.0)
```

## 📝 Files Description

### Core Notebooks
- **`00_eda.ipynb`**: Comprehensive exploratory data analysis
- **`03_impression_model_ft.ipynb`**: Complete fine-tuning pipeline with training metrics
- **`04_model_evaluation.ipynb`**: Systematic evaluation across modalities

### Data Processing
- **`enhanced_preprocessing.py`**: Findings-focused preprocessing pipeline
- **`analyze_filtering.py`**: Data quality analysis and filtering validation

### Documentation
- **`report/report.tex`**: Complete technical assessment report with methodology, results, and analysis

## 🛠️ Development Notes

### Training Environment
- **Platform**: RunPod cloud instance
- **GPU**: RTX 4090 (24GB VRAM)
- **Training Time**: ~2 hours for full pipeline
- **Budget**: <$2 total cost

### Key Dependencies
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
evaluate>=0.4.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
```

## 📋 Future Improvements

1. **Output Parsing**: Enhanced impression extraction from model responses
2. **Template Refinement**: Optimized chat templates for cleaner outputs
3. **Style Consistency**: Improved detection of clinical formatting patterns
4. **Multi-Modal Integration**: Incorporate image data alongside text
5. **Scaling Strategies**: Unified models with modality-aware generation

## 📄 License

This project is for educational and research purposes. Please ensure compliance with healthcare data regulations and institutional policies when using with real medical data.

## 👤 Author

**Sabber Ahamed**
Applied Scientist Technical Assessment - Sirona Medical

## 🔗 Links

- [Fine-tuned Model](https://huggingface.co/sabber/medphi-radiology-summary-adapter)
- [Technical Report](report/report.tex)
- [Base Model](https://huggingface.co/microsoft/MediPhi-Instruct)

---

*This project demonstrates practical application of large language models in healthcare, focusing on automated radiology impression generation with significant performance improvements over baseline approaches.*