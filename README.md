# Credit Card Fraud Detection Using Anomaly Detection Techniques

## Objective:
#### The main goal of this project is to detect fraudulent credit card transactions using unsupervised anomaly detection methods. Due to the highly imbalanced nature of the dataset—where fraudulent transactions are extremely rare—traditional classification models are often ineffective. Instead, this project applies Isolation Forest and One-Class SVM to identify anomalies (frauds).

## Dataset Overview:
#### - The dataset consists of credit card transactions with features transformed using PCA for confidentiality, except for Time and Amount.

#### - The target variable is Class (0: Non-fraudulent, 1: Fraudulent).

#### - The dataset is highly imbalanced, which is visualized using a count plot.

## Exploratory Data Analysis (EDA):
#### - info(), describe(), and value_counts() were used to inspect the data structure and class distribution.

#### - A bar plot was generated to show the imbalance between normal and fraud transactions.

## Data Preprocessing:
#### - StandardScaler was used to normalize the Amount feature.

#### - The Time and original Amount columns were dropped.

#### - PCA was applied to reduce the data to 2 dimensions for visualization. A scatter plot was created to show the separation between classes in reduced dimensions.

## Anomaly Detection Techniques:
### - Isolation Forest:

#### -- Assumes a small contamination ratio (0.17%) to reflect fraud rarity.

#### -- Converts anomaly output (-1 for outliers) into binary classification (1: fraud, 0: normal).

#### -- Outputs a classification report (precision, recall, f1-score, etc.).

### - One-Class SVM:

#### -- Trained on all data assuming the majority is normal.

#### -- Uses RBF kernel with nu=0.0017.

#### -- Outputs another classification report.

### - Evaluation: Precision-Recall Curve:
#### -- A custom function is used to plot Precision-Recall curves for both models.

#### -- Area Under Curve (AUC) scores are included to evaluate model performance on imbalanced data.

### - Dimensionality Reduction with PCA:
#### -- PCA was applied again with 10 components to compress the feature space.

#### -- Isolation Forest was re-applied on the PCA-reduced data.

#### -- A new classification report is generated for the model using PCA + Isolation Forest.

## Conclusion:
#### This project demonstrates the effectiveness of unsupervised anomaly detection (Isolation Forest and One-Class SVM) in detecting rare fraud cases in a highly imbalanced credit card dataset. It also highlights the usefulness of PCA for visualization and dimensionality reduction, and emphasizes the importance of Precision-Recall analysis for evaluating performance on imbalanced datasets.
