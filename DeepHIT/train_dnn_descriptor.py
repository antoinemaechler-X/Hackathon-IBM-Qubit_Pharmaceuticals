import os
import torch
import utils
import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple

class DescriptorDataset(Dataset):
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # Reshape to (batch_size, 1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def train_dnn_with_dataloader(X_train, y_train, model, batch_size=32, learning_rate=0.001, epochs=100, save_path="descriptor_based_dnn.pth"):
    """
    Train a DNN model using a DataLoader with tqdm progress bars.

    :param X_train: Training feature data.
    :param y_train: Training labels.
    :param model: PyTorch model to be trained.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param epochs: Number of training epochs.
    :param save_path: Path to save the trained model.
    :return: Trained PyTorch model.
    """
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Create DataLoader
    dataset = DescriptorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with tqdm progress bar
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for inputs, labels in epoch_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            y_pred = model(inputs)
            
            # Compute loss
            loss = criterion(y_pred, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            

        # print bar every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
            train_acc = np.mean(predict_dnn(model, X_train) == y_train)
            print(f"Train Accuracy: {train_acc:.4f}")
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model trained and saved at {save_path}")

    return model

def predict_dnn(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    """
    Predict using a trained PyTorch model.
    
    Args:
        model: Trained PyTorch model.
        X_test: Test features.
        
    Returns:
        Binary predictions as a NumPy array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).squeeze()
        return y_pred.round().cpu().numpy()

if __name__ == '__main__':
    # Load data
    data = pd.read_csv("data/train.csv")
    X, smiles_list = utils.extract_selected_features(data)
    y = data['class'].to_numpy()
    X = X.to_numpy()

    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.2, random_state=42)

    # Model configuration
    config = {
        "input_size": X_train.shape[1],
        "hidden_nodes": 892,
        "hidden_layers": 4,
        "dropout_rate": 0.3,
        "learning_rate": 0.0001,
        "epochs": 100,
        "batch_size": 32,
        "save_path": "DeepHIT/weights/descriptor_based_dnn.pth"
    }

    # Initialize and train model
    model = models.DescriptorBasedDNN(
        config["input_size"],
        config["hidden_nodes"],
        config["hidden_layers"],
        config["dropout_rate"]
    )
    train_params = {key: config[key] for key in ["batch_size", "learning_rate", "epochs", "save_path"]}
    model = train_dnn_with_dataloader(X_train, y_train, model, **train_params)
    # Predict on test set
    y_pred = predict_dnn(model, X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")
