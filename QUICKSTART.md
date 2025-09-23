# Quick Start Guide

## 🚀 Getting Started

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/fuzzy-gospel-consolidation.git
cd fuzzy-gospel-consolidation
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python test_basic.py
```

### 4. Run Basic Test

```bash
# Test data loading only
python src/main.py --mode export --data-dir data/raw --output-dir results/test
```

### 5. Run Full Pipeline

```bash
# Run complete pipeline
python src/main.py --config config.yaml --verbose
```

## 📁 Project Structure

```
fuzzy-gospel-consolidation/
├── src/                          # Source code
│   ├── data_processing/          # XML parsing and text extraction
│   ├── fuzzy_relations/          # Fuzzy logic calculations
│   ├── graph_neural_network/     # GNN implementation
│   ├── summarization/            # Text generation
│   ├── evaluation/               # Assessment metrics
│   └── main.py                   # Main pipeline
├── data/
│   └── raw/                      # XML input files (included)
├── results/                      # Output directory
├── config.yaml                   # Configuration file
└── requirements.txt              # Dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model parameters (GNN layers, hidden dimensions)
- Fuzzy logic weights and thresholds
- Summarization settings
- Evaluation metrics

## 📊 Expected Output

The pipeline generates:

1. **Consolidated Summary** (`results/consolidated_summary.txt`)
2. **Evaluation Report** (`results/evaluation_results.json`)
3. **Statistics** (`results/statistics.json`)
4. **Intermediate Data** (fuzzy relations, graph data)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install sentence-transformers torch torch-geometric
   ```

2. **CUDA Issues**: Set device to "cpu" in config.yaml if no GPU
   ```yaml
   models:
     device: "cpu"
   ```

3. **Memory Issues**: Reduce batch size or hidden dimensions in config

### Getting Help

- Check logs in `logs/fuzzy_gospel.log`
- Run with `--verbose` flag for detailed output
- Verify data files are in `data/raw/`

## 📈 Next Steps

1. **Experiment with Parameters**: Modify fuzzy weights in config.yaml
2. **Add Custom Metrics**: Extend evaluation module
3. **Visualize Results**: Use graph data for network visualization
4. **Scale Up**: Process additional Gospel sections

## 🎯 For Research

This project is designed for the IEEE WCCI 2026 submission. Key research contributions:

- **Fuzzy Temporal Anchoring**: Novel approach to temporal uncertainty
- **Graph-Enhanced Consolidation**: GNN-based narrative merging
- **Comprehensive Evaluation**: Multi-dimensional assessment framework

Happy researching! 🔬
