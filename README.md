# ProteinLLMs: Predicting Kinase ATP Binding Sites with ESM

Protein language model project using ESM-2 embeddings to analyze kinases and predict ATP binding sites. The pipeline compares pooling strategies, evaluates clustering quality, applies E2EATP for residue-level predictions, and assesses performance with classification metrics and ROC analysis.

## Project Goals

- Generate high-quality protein embeddings from kinase sequences using ESM-2
- Evaluate pooling strategies for protein-level representations
- Predict ATP binding residues using residue-level embeddings
- Quantitatively assess prediction performance and visualize results

## Methods Overview

### Protein Embeddings with ESM-2
- Parsed kinase sequences from FASTA files (UniProt format)
- Generated per-residue embeddings using the ESM2-T33 model
- Removed start/stop tokens to preserve true sequence length
- Saved embeddings for reuse and downstream tasks

### Representation Pooling and Analysis
- Applied mean pooling and max pooling to obtain protein-level vectors
- Visualized embeddings with t-SNE
- Evaluated clustering quality using silhouette scores based on protein family labels
- Compared how pooling strategies affect family separation

### ATP Binding Site Prediction
- Used residue-level embeddings as input to the E2EATP model
- Predicted ATP binding probabilities for each amino acid in each kinase
- Aggregated predictions using probability cutoffs
- Analyzed amino acid frequencies among predicted binding sites

### Model Evaluation
- Compared predictions against known ATP binding residues
- Computed accuracy, precision, true positive rate, and false positive rate
- Evaluated performance across multiple thresholds
- Generated ROC curves to assess classifier behavior

### Structural Visualization
- Mapped predicted binding residues onto kinase 3D structures
- Visualized results using PyMOL to interpret spatial organization

## Key Scripts

- **esm_model.py**  
  Loads FASTA sequences, generates ESM embeddings, applies pooling, and produces visualizations.

- **analyze.py**  
  Analyzes E2EATP predictions, computes metrics, plots histograms, and generates ROC curves.

- **predict.py** (provided)  
  Runs E2EATP inference using residue-level embeddings.

## Outputs

- Per-residue and pooled protein embeddings
- t-SNE visualizations of protein families
- Silhouette score comparisons for pooling strategies
- ATP binding site prediction files
- Performance metrics and ROC curves
- Structural visualizations of predicted binding sites

## Technologies Used

- Python 3.12.1
- PyTorch
- ESM / fair-esm
- Biopython
- NumPy / Pandas
- scikit-learn
- Matplotlib
- PyMOL

