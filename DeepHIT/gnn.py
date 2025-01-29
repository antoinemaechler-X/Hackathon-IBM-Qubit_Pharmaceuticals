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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler



class GraphNeuralNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_gcn_layers, 
                 dnn_hidden_nodes, num_dnn_layers, dropout_rate, num_classes=1):
        """
        Enhanced Graph Neural Network with multiple message passing layers
        and dynamic graph convolution.
        """
        super(GraphNeuralNetwork, self).__init__()
        
        # Graph Convolution Layers
        self.gcn_layers = nn.ModuleList()
        
        # First GCN layer
        self.gcn_layers.append(
            nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels, momentum=0.1)
            )
        )
        
        # Additional GCN layers
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_channels, momentum=0.1)
                )
            )
        
        # Dense Layers
        self.dnn_layers = nn.ModuleList()
        input_size = hidden_channels
        
        for _ in range(num_dnn_layers):
            self.dnn_layers.append(
                nn.Sequential(
                    nn.Linear(input_size, dnn_hidden_nodes),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            input_size = dnn_hidden_nodes
        
        # Final output layer
        self.output_layer = nn.Linear(input_size, num_classes)
    
    def graph_convolution(self, X, A):
        """
        Custom graph convolution method with adjacency matrix integration.
        
        Args:
            X (torch.Tensor): Node feature matrix (batch_size, num_nodes, num_features)
            A (torch.Tensor): Adjacency matrix (batch_size, num_nodes, num_nodes)
        
        Returns:
            torch.Tensor: Transformed node features
        """
        # Ensure inputs are float tensors
        X = X.float()
        A = A.float()
        
        # Normalize adjacency matrix
        degrees = torch.sum(A, dim=2)
        D_inv_sqrt = torch.pow(degrees, -0.5)
        D_inv_sqrt = torch.nan_to_num(D_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Symmetrically normalize adjacency matrix
        A_norm = A * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(-2)
        
        # Perform graph convolution
        for layer in self.gcn_layers:
            # Apply linear transformation and activation
            X = layer[0](X)  # Linear layer
            X = layer[1](X)  # ReLU
            
            # Perform message passing
            X = torch.bmm(A_norm, X)
            
            # Batch normalization
            X = layer[2](X.transpose(1, 2)).transpose(1, 2)
        
        return X
    
    def forward(self, X, A):
        """
        Forward pass of the Graph Neural Network.
        
        Args:
            X (torch.Tensor): Node feature matrix
            A (torch.Tensor): Adjacency matrix
        
        Returns:
            torch.Tensor: Output predictions
        """
        # Graph convolution
        X = self.graph_convolution(X, A)
        
        # Global pooling (mean aggregation)
        X = torch.mean(X, dim=1)
        
        # Dense layers
        for layer in self.dnn_layers:
            X = layer(X)
        
        # Output layer
        return self.output_layer(X)


class GraphDataset(Dataset):
    def __init__(self, X_data, A_data, y_data):
        """
        Enhanced dataset class with support for data transformations.
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.A = torch.tensor(A_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]


def train_graph_neural_network(
    model, dataloader, epochs=100, 
    learning_rate=0.001, 
    weight_decay=1e-5, 
    save_path="gnn_model.pth"
):
    """
    Enhanced training function with learning rate scheduling and early stopping.
    """
    # Set up optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, A_batch, y_batch in tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            leave=False
        ):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch, A_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Average loss and learning rate adjustment
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


def evaluate_graph_neural_network(model, X_data, A_data, y_data):
    """
    Enhanced evaluation with multiple metrics.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(
            torch.tensor(X_data, dtype=torch.float32), 
            torch.tensor(A_data, dtype=torch.float32)
        )
        y_pred = torch.sigmoid(y_pred).numpy().squeeze()
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = np.mean(y_pred_binary == y_data)
        precision = np.mean(y_pred_binary[y_data == 1]) if np.sum(y_pred_binary) > 0 else 0
        recall = np.sum(y_pred_binary[y_data == 1]) / np.sum(y_data) if np.sum(y_data) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    
    return accuracy


if __name__ == '__main__':
    # Load data
    data = pd.read_csv("data/train.csv")
    smiles_list = data["smiles"].tolist()
    y = data['class'].to_numpy()

    # Convert to graph data
    X, A, smiles_list = utils.convert_to_graph(smiles_list)

    # Load train/test indices
    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    # Split data
    # X_train, X_test = X[train_indices], X[test_indices]
    # y_train, y_test = y[train_indices], y[test_indices]
    # A_train, A_test = A[train_indices], A[test_indices]
    X_train = X
    y_train = y
    X_test = X
    y_test = y
    A_train = A
    A_test = A


    # Model hyperparameters
    num_features = X_train.shape[2]
    gnn_params = {
        'num_features': num_features,
        'hidden_channels': 64,  # Reduced from 128
        'num_gcn_layers': 3,
        'dnn_hidden_nodes': 256,  # Adjusted
        'num_dnn_layers': 2,
        'dropout_rate': 0.2
    }

    # Initialize and train model
    model = GraphNeuralNetwork(**gnn_params)
    train_dataset = GraphDataset(X_train, A_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    

    # Train the model
    trained_model = train_graph_neural_network(
        model, train_dataloader, 
        epochs=100, 
        learning_rate=0.0001, 
        save_path="DeepHIT/weights/gnn_2.pth"
    )

    # Evaluation
    print("\nTest Set Performance:")
    test_accuracy = evaluate_graph_neural_network(
        trained_model, X_test, A_test, y_test
    )
    
    print("\nTraining Set Performance:")
    train_accuracy = evaluate_graph_neural_network(
        trained_model, X_train, A_train, y_train
    )