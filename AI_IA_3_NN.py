# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/17eRavA2tm1JkNaUmWh2w002DaoCB22AS
"""

!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=350)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

column_names = [
    "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
    "default"
]

df.columns = column_names
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

df.info()
df['default'].value_counts()

import numpy as np

# Handle NaN + infinite
df = df.fillna(0)
df.replace([np.inf, -np.inf], 0, inplace=True)

df['default'] = df['default'].astype(int)

bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
pay_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

# Financial behavior features
df['avg_bill'] = df[bill_cols].mean(axis=1)
df['avg_payment'] = df[pay_cols].mean(axis=1)

df['payment_ratio'] = df['avg_payment'] / (df['avg_bill'] + 1)

# Risk features
df['max_delay'] = df[["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].max(axis=1)

df['utilization'] = df['avg_bill'] / (df['LIMIT_BAL'] + 1)

df['LIMIT_BAL'] = np.log1p(df['LIMIT_BAL'])
df['avg_bill'] = np.log1p(df['avg_bill'])
df['avg_payment'] = np.log1p(df['avg_payment'])

from sklearn.model_selection import train_test_split

X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import numpy as np

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

X_train = X_train.clip(-1e6, 1e6)
X_test = X_test.clip(-1e6, 1e6)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print(class_weights)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy Curve")
plt.show()

y_pred_prob = model.predict(X_test_scaled).flatten()


y_pred_prob = np.nan_to_num(y_pred_prob)

y_pred = (y_pred_prob > 0.3).astype(int)

from sklearn.metrics import classification_report, roc_auc_score

y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

import numpy as np


X_full = df.drop(columns=['default']).copy()


X_full = X_full.astype(np.float64)


X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(0)


X_full = X_full.clip(-1e6, 1e6)


X_full_scaled = scaler.transform(X_full)


df['PD'] = model.predict(X_full_scaled).flatten()

sample = X_test.iloc[:5]

sample_scaled = scaler.transform(sample)

pred_prob = model.predict(sample_scaled)

for i, prob in enumerate(pred_prob):
    print(f"Client {i+1} Default Probability: {prob[0]:.2f}")

model.save("credit_model.h5")

!pip install pyngrok

import netron

netron.start("credit_model.h5", address=("0.0.0.0", 8081))

from google.colab import files
files.download("credit_model.h5")