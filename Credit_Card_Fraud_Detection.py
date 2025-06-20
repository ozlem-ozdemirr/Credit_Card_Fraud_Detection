import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, precision_recall_curve, auc
import warningswarnings.filterwarnings("ignore")


df = pd.read_csv("creditcard.csv")df.head()

print(df.info())
print(df.describe())
print("Class Distribution:\n", df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Imbalanced Data Distribution (0: Normal, 1: Fraud)")
plt.show()

scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df.drop('Class', axis=1))
pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
pca_df['Class'] = df['Class']
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Class', palette={0:'blue',1:'red'})
plt.title("Visualization with PCA")
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']
iso_forest = IsolationForest(contamination=0.0017, random_state=42)
iso_pred = iso_forest.fit_predict(X)
iso_pred = np.where(iso_pred == 1, 0, 1)
print("Isolation Forest Classification Report")
print(classification_report(y, iso_pred))

oc_svm = OneClassSVM(nu=0.0017, kernel="rbf", gamma='scale')
oc_svm.fit(X)svm_pred = oc_svm.predict(X)
svm_pred = np.where(svm_pred == 1, 0, 1)
print("One-Class SVM Classification Report")
print(classification_report(y, svm_pred))


def plot_precision_recall(y_true, y_scores, title):    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)    
    pr_auc = auc(recall, precision)    
    plt.plot(recall, precision, marker='.', label=f'{title} (AUC={pr_auc:.4f})')    
    plt.xlabel('Recall')    
    plt.ylabel('Precision')    
    plt.title('Precision-Recall Curve')    
    plt.legend()    
    plt.grid(True)
    plt.figure(figsize=(10,5))
    plot_precision_recall(y, iso_pred, "Isolation Forest")
    plot_precision_recall(y, svm_pred, "One-Class SVM")
    plt.show()


pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
iso_pca = IsolationForest(contamination=0.0017, random_state=42)
iso_pca.fit(X_pca)
iso_pred_pca = np.where(iso_pca.predict(X_pca) == 1, 0, 1)
print("Isolation Forest + PCA Classification Report")
print(classification_report(y, iso_pred_pca))
