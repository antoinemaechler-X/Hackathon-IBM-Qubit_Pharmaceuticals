import models
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DescriptorDataset(Dataset):
    def __init__(self, X_data, A_data, y_data):
        """
        Dataset class to handle graph data for GCN.
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.A = torch.tensor(A_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]


def train_gcn(model, dataloader, epochs=100, learning_rate=0.01, save_path="gcn.pth"):
    """
    Function to train the Graph Neural Network (GNN) model.

    Args:
        model: The GCN model to be trained.
        dataloader: DataLoader for the training data.
        epochs: Number of epochs for training.
        learning_rate: Learning rate for the optimizer.
        save_path: Path to save the trained model weights.

    Returns:
        The trained model.
    """
    # Set the model to training mode
    model.train()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, A_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch, A_batch)

            # Compute loss
            loss = criterion(y_pred, y_batch)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log the average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_gcn(model, X_data, A_data, y_data):
    """
    Function to evaluate the GCN model.

    Args:
        model: The trained GCN model.
        X_data: Feature data.
        A_data: Adjacency matrices.
        y_data: Ground truth labels.

    Returns:
        Accuracy of the model on the given data.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_data, dtype=torch.float32), torch.tensor(A_data, dtype=torch.float32))
        y_pred = torch.sigmoid(y_pred).numpy().squeeze()
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_data)
    return accuracy


if __name__ == '__main__':
    # Load data
    data = pd.read_csv("data/train.csv")
    smiles_list = data["smiles"].tolist()
    y = data['class'].to_numpy()

    # Convert to graph data
    X, A, smiles_list = utils.convert_to_graph(smiles_list)

    print(f"X shape: {X.shape}, A shape: {A.shape}, y length: {len(y)}")

    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    A_train, A_test = A[train_indices], A[test_indices]
    # Split the data
    
    # Print dataset shapes
    print(f"Train X shape: {X_train.shape}, Train A shape: {A_train.shape}")
    print(f"Test X shape: {X_test.shape}, Test A shape: {A_test.shape}")

    # Model parameters
    num_features = X_train.shape[2]
    gcn_hidden_nodes = 64
    gcn_hidden_layers = 3
    dnn_hidden_nodes = 1024
    dnn_hidden_layers = 2
    dropout_rate = 0.1

    # Initialize the GCN model
    model = models.GCN_Model(num_features, gcn_hidden_nodes, gcn_hidden_layers, dnn_hidden_nodes, dnn_hidden_layers, dropout_rate)

    # Prepare DataLoader
    train_dataset = DescriptorDataset(X_train, A_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Train the model
    trained_model = train_gcn(model, train_dataloader, epochs=100, learning_rate=0.0001, save_path="gcn.pth")

    # Load the trained model for evaluation
    model.load_state_dict(torch.load("gcn.pth"))

    # Evaluate the model
    test_accuracy = evaluate_gcn(model, X_test, A_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    train_accuracy = evaluate_gcn(model, X_train, A_train, y_train)
    print(f"Train Accuracy: {train_accuracy:.4f}")
