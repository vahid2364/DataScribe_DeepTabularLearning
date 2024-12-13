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

This project addresses the challenges in tabular materials science data, including extreme skewness, multimodal distributions, and complex feature interactions. The repository implements several machine learning models, ranging from classical tree-based methods (e.g., XGBoost) to advanced deep learning architectures (e.g., encoder-decoder models, TabNet, and Variational Autoencoders).

Key features of the project:
- **Tabular data preprocessing** for skewness and outliers.
- **Benchmarking of deep learning models** (e.g., Fully Dense Networks, Disjunctive Normal Form Networks, TabNet).
- **Bayesian optimization** for hyperparameter tuning.
- Comparative analysis of predictive performance and computational efficiency.

---

## 📂 Repository Structure

```
.
├── data/
│   ├── raw/                       # Raw input data files
│   ├── processed/                 # Preprocessed data files
│   └── README.md                  # Details about datasets
├── models/
│   ├── encoder_decoder.py         # Encoder-Decoder implementation
│   │          ├── Fully Dense architecture
│   │          ├── DNNF architecture
│   │          └── 1D-CNN architecture
│   │          
│   ├── tabnet.py                  # TabNet model
│   ├── vae.py                     # Variational Autoencoder implementation
│   ├── xgboost_baseline.py        # XGBoost baseline model
│   └── utils.py                   # Utility functions for training and evaluation
├── results/
│   ├── figures/                   # Figures and plots
│   ├── tables/                    # Results tables (e.g., metrics, hyperparameters)
│   └── README.md                  # Description of results
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Exploratory Data Analysis (EDA)
│   ├── 02_model_training.ipynb    # Model training and evaluation
│   ├── 03_hyperparameter_tuning.ipynb  # Hyperparameter optimization using TPE
│   └── 04_results_visualization.ipynb  # Visualization of model performance
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
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

### Description
The dataset includes tabular materials data with features such as:
- **Compositional information**: Nb, Cr, V, W, Zr.
- **Material properties**: Thermal conductivity, density, yield strength, creep rate.

### Preprocessing
1. Missing values are handled via interpolation.
2. Features are normalized using quantile transformation for skewed distributions.

For more details, see the [data README](data/README.md).

---

## 🚀 Models

The repository implements and benchmarks the following models:
- **XGBoost**: Baseline tree-based model.
- **Fully Dense Neural Network (FDN)**: Standard fully connected architecture.
- **Disjunctive Normal Form Network (DNF-Net)**: Captures logical relationships in tabular data.
- **TabNet**: Attention-based feature selection for interpretability.
- **Variational Autoencoder (VAE)**: Generative model for handling uncertainty.

### Key Metrics
- **Mean Squared Error (MSE)**
- **R-squared (R²)**
- **Symmetric Mean Absolute Percentage Error (SMAPE)**

See [models README](models/README.md) for implementation details.

---

## 🔍 Results

1. **Performance Comparison**:
   - XGBoost achieves the best computational efficiency and strong predictive performance.
   - DNF-Net outperforms on skewed features but has higher computational costs.

2. **Optimization Insights**:
   - Bayesian optimization using Tree-structured Parzen Estimator (TPE) fine-tunes hyperparameters.
   - Scaling and quantile transformations significantly improve model stability.

Key results are summarized in the [results directory](results/).

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

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## 📞 Contact

For questions or issues, contact the authors:
- Vahid Attari: attari.v@tamu.edu


