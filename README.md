# GACET: Graph-Aware Cross-domain EEG Transformer 

## Environment
- Python 3.12.6  
- PyTorch 2.4.0  
- CUDA 12.1  
- Hardware: Dual NVIDIA GeForce RTX 3090 GPUs  

## Project Structure

GACET/
├── data/ # single-subject data for running train_model.ipynb
├── feature/ # DE, PSD & SampEn feature extraction implementations
├── pre_processing/ # data pre-processing scripts
├── train/ # model definitions & required runtime files
├── log/ # significance analysis & cross-day evaluation logs
└── train_model.ipynb # Jupyter notebook to train and evaluate the model
