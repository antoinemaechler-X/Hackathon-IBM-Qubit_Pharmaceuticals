# Save indices
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv("data/train.csv")
train_indices, test_indices = train_test_split(
    np.arange(len(data)), test_size=0.2, random_state=42, stratify=data['class']
)
np.save("train_indices.npy", train_indices)
np.save("test_indices.npy", test_indices)

# Load indices
train_indices = np.load("train_indices.npy")
test_indices = np.load("test_indices.npy")

X_train_raw, X_test_raw = data.iloc[train_indices], data.iloc[test_indices]
y_train, y_test = data['class'].iloc[train_indices], data['class'].iloc[test_indices]
