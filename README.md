# Fuzzy Gospel Consolidation

**Fuzzy Temporal Anchoring for Narrative Consolidation in Multi-Document Summarization**

A novel approach to consolidating multiple narrative accounts using fuzzy logic and Graph Neural Networks, demonstrated on the four canonical Gospels Holy Week accounts.

##  Project Overview

This project implements a fuzzy-enhanced Graph Neural Network (Fuzzy-GNN) framework for consolidating multiple overlapping narrative documents. The system addresses three key challenges in multi-document summarization:

1. **Temporal Anchoring**: Using fuzzy logic to handle uncertainty in event timing and sequence
2. **Version Consolidation**: Intelligently merging conflicting accounts from multiple sources  
3. **Narrative Coherence**: Maintaining chronological flow while preserving important details

The system successfully processes 144 distinct Gospel events, calculating 10,296 fuzzy relations and achieving exceptional performance in narrative consolidation with 86.98% overall evaluation score.

##  Dataset

The project uses the Holy Week accounts from the four canonical Gospels:
- **144 distinct events** from the chronology file
- **4 Gospel texts** (Matthew, Mark, Luke, John) in XML format
- **Precise verse references** for automatic text extraction
- **Temporal annotations** (days of the week, sequence information)

##  Architecture

```
Input: 4 Gospel XMLs + Chronology  Fuzzy Relations  GNN  Consolidated Summary
```

### Key Components:

1. **Data Processing**: XML parsing and text extraction
2. **Fuzzy Relations**: Calculate membership degrees for event relationships
3. **Graph Neural Network**: Process fuzzy-enhanced event graph
4. **Summarization**: Generate consolidated narrative using Transformer models
5. **Evaluation**: Multi-dimensional assessment of output quality

##  Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM
- Virtual environment support

### Installation

```bash
git clone https://github.com/neemias8/Fuzzy-Gospel-Consolidation.git
cd Fuzzy-Gospel-Consolidation

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the complete pipeline
python src/main.py

# The system will automatically:
# 1. Parse XML Gospel files
# 2. Calculate fuzzy relations (10,296 relations)
# 3. Train Fuzzy-GNN model (144 nodes, 20,500 edges)
# 4. Generate consolidated summary
# 5. Evaluate against Golden Sample reference
```

##  Project Structure

```
fuzzy-gospel-consolidation/
 src/
    data_processing/          # XML parsing and text extraction
    fuzzy_relations/          # Fuzzy membership calculations
    graph_neural_network/     # GNN implementation
    summarization/            # Text generation and consolidation
    evaluation/               # Assessment metrics and protocols
    main.py                   # Main pipeline orchestrator
 data/
    raw/                      # Original XML files
    processed/                # Processed datasets
 results/                      # Output summaries and evaluations
 tests/                        # Unit and integration tests
 requirements.txt              # Python dependencies
 README.md                     # This file
```

##  Results

The system demonstrates exceptional performance in Gospel consolidation:

### Automatic Metrics (vs. Golden Sample Reference)
- **ROUGE-1**: 95.22% (content overlap)
- **ROUGE-2**: 92.70% (bigram overlap) 
- **ROUGE-L**: 91.44% (longest common subsequence)
- **BERTScore F1**: 90.27% (semantic similarity)
- **METEOR**: 91.40% (alignment-based metric)
- **BLEU**: 88.54% (n-gram precision)

### Temporal Coherence
- **Kendall Tau**: 100.00% (perfect chronological ordering)
- **Event Coverage**: 141/144 events (97.9%) mentioned in final summary

### System Performance
- **Processing**: 10,296 fuzzy relations calculated
- **Model Convergence**: Loss reduced to 0.0010 in 100 epochs
- **Overall Score**: 86.98% comprehensive evaluation score

##  Development

### Running Tests

```bash
python -m pytest tests/ -v
python test_basic.py  # Basic functionality test
```

### Understanding the Pipeline

The system follows this execution flow:
1. **Data Loading**: Parse XML Gospel files and chronology
2. **Fuzzy Relations**: Calculate 4 membership functions per event pair
3. **Graph Construction**: Build fuzzy-enhanced graph (144 nodes, 20,500 edges)
4. **GNN Training**: Self-supervised learning with convergence monitoring
5. **Summarization**: Generate consolidated narrative using BART-large-cnn
6. **Evaluation**: Comprehensive assessment against Golden Sample reference

### Output Files

After execution, check the results/ directory for:
- consolidated_summary_[timestamp].txt - Generated narrative
- evaluation_results_[timestamp].json - Detailed metrics
- evaluation_report_[timestamp].txt - Human-readable report

##  Citation

**IF ACCEPTED AT THE TARGET CONFERENCE**: if you use this work in your research, please cite:

```bibtex 
@inproceedings{finger2026fuzzy,
  title={Fuzzy Temporal Anchoring for Narrative Consolidation in Multi-Document Summarization},
  author={Finger, Roger Antonio and Ramos, Gabriel de Oliveira},
  booktitle={2026 IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026},
  organization={IEEE},
  note={PhD Thesis Research - UNISINOS Applied Computing Program}
}
```

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Contact

- **Author**: Roger Antonio Finger
- **Advisor**: Prof. Dr. Gabriel de Oliveira Ramos
- **Institution**: UNISINOS - Programa de Pós-Graduação em Computação Aplicada

---

*This project is part of a PhD thesis in Applied Computing at UNISINOS, focusing on temporal anchoring and version consolidation in abstractive multi-document summarization.*
