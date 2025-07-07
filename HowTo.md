# Deep AntiPhish
A deep learning-based system designed to detect social engineering attacks, particularly phishing and spear-phishing emails, using a rich combination of Natural Language Processing (NLP), metadata analysis, and deep neural networks. This project was developed as an academic research initiative to improve email security using modern machine learning techniques.

## Motivation
Phishing remains one of the most dominant cyberattack vectors with increasing sophistication. Traditional rule-based filters often fail to keep up with the evolving techniques of attackers. The motivation behind DeepAntiPhish is to build a robust system that can analyze both the **email content** and **metadata** to detect malicious emails with high accuracy.

## Scope of this project
- Parses raw email files (EML and MBOX)
- Extracts metadata: sender details, headers (`Subject`, `Reply-To`, `Return-Path`, etc.), URLs, and attachments
- Performs feature engineering and vectorization
- Trains a **deep neural network** to classify emails as **phishing** or **safe**
- Evaluates performance using metrics like **precision**, **recall**, **F1-score**, and **confusion matrix**
- Visualizes model insights: correlation matrix, feature importance, and error analysis

## System Design & Implementation
The pipeline includes the following:

### 1. **Data Collection**
- Datasets used: SpamAssassin, Phishing Corpus, and Enron MBOX dataset
- Split into Train and Test:
  - Train: 5302 safe + 6048 phishing
  - Test: 1900 safe + 1897 phishing

### 2. **Feature Engineering**
- Metadata features: `name`, `username`, `mail_domain`, `return_path`, `x_mailer`, etc.
- Content features: `body_text`, `url`, `subject`, `received`
- Numerical features: `body_length`, `url_count`, `attachment_count`, etc.
- URL/attachment parsing to handle multiple rows per email for detailed analysis

### 3. **Model Architecture**
A 7-layer feedforward deep neural network:
- Input layer → 5 Hidden layers (with ReLU + Dropout) → Output layer with sigmoid
- Training strategy: 10-epoch cycles, resume from best model checkpoint
- Final model: 98.10% validation accuracy

## Visualizations
- Correlation Matrix (Feature correlation to label)
- Mutual Information Scores (Non-linear dependencies)
- ROC Curve, AUC Score
- Precision vs Recall
- Error Analysis: False Positives vs False Negatives
- Feature Behavior of misclassified samples using parallel coordinates

## Results

| Metric       | Score      |
|--------------|------------|
| Accuracy     | 99.56%     |
| Precision    | 1.0000     |
| Recall       | 0.9945     |
| F1-Score     | 0.9972     |
| Test Samples | 53,685     |


## Technologies Used

- Python 3.12
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Seaborn, Matplotlib
- Jupyter Notebooks

## Academic Status

This project was developed as part of an academic research course on secure systems and machine learning. It demonstrates the end-to-end design of a cybersecurity AI solution, from raw data parsing to intelligent decision-making.

## Contributors
**Shruti Singh**
