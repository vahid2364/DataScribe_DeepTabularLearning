<div align="center">
   
   # DataScribe Tabular Learning Models
   
   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit/)
   ![Python](https://img.shields.io/badge/python-3.11%2B-blue)
   ![Optuna](https://img.shields.io/badge/Optuna-4.0-brightgreen)
   ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange?logo=tensorflow)
   ![Keras](https://img.shields.io/badge/Keras-3.9.2-red?logo=keras)
   ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-f7931e?logo=scikit-learn)
   ![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)
   
   <p>
      <a href="https://github.com/vahid2364/DataScribe_DeepTabularLearning/issues/new">Report a Bug</a> |
      <a href="https://github.com/vahid2364/DataScribe_DeepTabularLearning/issues/new?labels=enhancement">Request a Feature</a>
   </p>
      
</div>

---

# DataScribe Deep Tabular Learning

This repository provides the code, data, and documentation for the paper:

**[Decoding Non-Linearity and Complexity: Deep Tabular Learning Approaches for Materials Science](https://arxiv.org/abs/2411.18717)**  
*Vahid Attari, Raymundo Arroyave*  
Department of Materials Science & Engineering, Texas A&M University.

<div align="center">
   <img src="https://media.licdn.com/dms/image/v2/C4E12AQE98dfpdYhxZA/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1610116669577?e=2147483647&v=beta&t=A7bKteJy6T0CUcImJUute3Bio5J_olkFKVXwgy-TMP8" alt="Work in Progress" width="200">
</div>

---

## 📖 Overview

This project addresses the challenges in tabular materials science data, including extreme skewness, and complex feature interactions. The repository implements several machine learning models, ranging from classical tree-based methods (e.g., XGBoost) to advanced deep learning architectures (e.g., encoder-decoder models, TabNet, and Variational Autoencoders) on various materials science datasets.

Key features of the project:
- **Tabular data preprocessing** for skewness and outliers.
- **Benchmarking of deep learning models** (e.g., regularized Dense Networks, Disjunctive Normal Form Networks, TabNet).
- **Bayesian optimization** for hyperparameter tuning.
- Comparative analysis of predictive performance and computational efficiency.

---

## 📂 Repository Structure

The repository currently includes multiple datasets, e.g., ATLAS-RHEA, BIRDSHOT, and MPEA. Each dataset folder contains several models along with their corresponding Bayesian hyperparameter optimization workflows and post-processing results. Below is an example folder structure:

```
.
├── datasets/
│   ├─── ATLAS-RHEA/
│   │    ├── data/
│   │    │      ├── raw/                       # Raw input data files
│   │    │      ├── processed/                 # Preprocessed data files
│   │    │      └── README.md                  # Details about datasets
│   │    │
│   │    ├── models/               # Trained models for ATLAS-RHEA
│   │    │      │
│   │    │      ├── encoder_decoder.py         # Encoder-Decoder implementation
│   │    │      │       ├── Fully Dense architecture
│   │    │      │       ├── DNNF architecture
│   │    │      │       └── 1D-CNN architecture
│   │    │      │    
│   │    │      ├── tabnet.py                  # TabNet model
│   │    │      ├── vae.py                     # Variational Autoencoder implementation
│   │    │      ├── xgboost_baseline.py        # XGBoost baseline model
│   │    │      └── utils.py                   # Utility functions for training and evaluation
│   │    │
│   │    ├── bayes_opt/             # Bayesian optimization scripts and results
│   │    └── postprocessing/        # Analysis and visualizations
│   │
│   ├── BIRDSHOT/
│   │   ├── data/
│   │   ├── models/
│   │   ├── bayes_opt/
│   │   └── postprocessing/
│   │
│   └── MPEA/
│       ├── data/
│       ├── models/
│       ├── bayes_opt/
│       └── postprocessing/
│
├── standalone software/
│   
│
├── README.md                      # Project overview and instructions
└── requirements.txt               # Python package requirements
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- A GPU is recommended for training deep learning models.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/vahid2364/DataScribe_DeepTabularLearning.git
   cd DataScribe_DeepTabularLearning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Datasets

### ATLAS-RHEA Description
The dataset includes tabular materials data with features such as:
- **Compositional information**: Nb, Cr, V, W, Zr.
- **Material properties**: Thermal conductivity, density, yield strength, creep rate.

### BIRDSHOT Description
The dataset includes tabular materials data with features such as:
- **Compositional information**: Al	Co	Cr	Cu	Fe	Mn	Ni	V (Bayesian and EDS measured).
- **Material properties**: Computed and experimentally evaluated mechanical properties including yield and ultimate tensile strength, hardness, depth of penetration, SFE, etc.

### BORG-HEA Description
The dataset includes tabular materials data with features such as:
- **Compositional information**: Multiple elements
- **Material properties**: Mechanical properties

### Preprocessing
1. Missing values are handled via interpolation.
2. Features are normalized using quantile transformation for skewed distributions.

---

## 🚀 Models

The repository implements and benchmarks the following models:
- **Encoder-Decoder (asymmetric overcomplete) regularized Dense Neural Network (rDN)**: Standard regularized dense architecture.
- **Encoder-Decoder Disjunctive Normal Form Network (DNF-Net)**: Captures logical relationships in tabular data.
- **Encoder-Decoder (1D-CNN)**: Extracts local patterns and hierarchical feature representations using 1D convolutional filters, suitable for structured tabular data with implicit ordering or grouped feature patterns..
- **TabNet**: Attention-based model that performs sequential feature selection, enabling interpretability while capturing complex feature interactions in tabular data.
- **Variational Autoencoder (VAE)**: Generative model for handling uncertainty.
- **XGBoost**: Baseline tree-based model.

### Key Metrics
- **Mean Squared Error (MSE)**
- **Mean Squared Logarithmic Error (MSLE)**
- **R-squared (R²)**
- **Symmetric Mean Absolute Percentage Error (SMAPE)**

---

## 🔍 Results

1. **Performance Comparison**:
   - XGBoost achieves the best computational efficiency and strong predictive performance.
   - DNF-Net outperforms on skewed features but has higher computational costs.
   - rDN model, when properly tuned, achieves strong performance on highly complex features.
   
2. **Optimization Insights**:
   - Bayesian optimization using Tree-structured Parzen Estimator (TPE) fine-tunes hyperparameters.
   - Scaling and quantile transformations significantly improve model stability when features are complex and span multiple orders of magnitude.

---

## 🧪 Usage

### Training a Model
Run the model training script:
```bash
python models/encoder_decoder.py --data data/processed/dataset.csv --epochs 50
```

### Visualizing Results
Use the provided Jupyter notebooks:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🛠️ Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Description of changes"`.
4. Push to your branch: `git push origin feature-name`.
5. Create a pull request.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📞 Contact

For questions or issues, contact the authors:
- Vahid Attari: attari.v@tamu.edu


