import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import utils

class GraphNeuralNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_gcn_layers, 
                 dnn_hidden_nodes, num_dnn_layers, dropout_rate, l2_lambda, num_classes=1):
        """
        Enhanced Graph Neural Network with adaptive batch normalization
        """
        super(GraphNeuralNetwork, self).__init__()
        
        # Regularization parameters
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        
        # Graph Convolution Layers
        self.gcn_layers = nn.ModuleList()
        
        # Input layer
        self.gcn_layers.append(
            nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels, momentum=0.1),
                nn.Dropout(dropout_rate)
            )
        )
        
        # Additional GCN layers
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_channels, momentum=0.1),
                    nn.Dropout(dropout_rate)
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
                    nn.BatchNorm1d(dnn_hidden_nodes, momentum=0.1),
                    nn.Dropout(dropout_rate)
                )
            )
            input_size = dnn_hidden_nodes
        
        # Final output layer
        self.output_layer = nn.Linear(input_size, num_classes)
    
    def graph_convolution(self, X, A):
        """
        Enhanced graph convolution with adjacency normalization
        """
        # Normalize adjacency matrix
        degrees = torch.sum(A, dim=2)
        D_inv_sqrt = torch.pow(degrees + 1e-7, -0.5)
        A_norm = A * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(-2)
        
        for layer in self.gcn_layers:
            # Linear transformation
            X = layer[0](X)
            
            # Message passing
            X = torch.bmm(A_norm, X)
            
            # Activation
            X = layer[1](X)
            
            # Batch normalization and dropout
            X = layer[2](X.transpose(1, 2)).transpose(1, 2)
            X = layer[3](X)
        
        return X
    
    def forward(self, X, A):
        """
        Forward pass with graph convolution and regularization
        """
        # Graph convolution
        X = self.graph_convolution(X, A)
        
        # Global pooling
        X = torch.mean(X, dim=1)
        
        # Dense layers
        for layer in self.dnn_layers:
            X = layer[0](X)  # Linear
            X = layer[1](X)  # ReLU
            X = layer[2](X)  # BatchNorm
            X = layer[3](X)  # Dropout
        
        # Output layer
        return self.output_layer(X)


class GraphDataset(Dataset):
    def __init__(self, X_data, A_data, y_data):
        """
        Dataset with tensor conversion
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.A = torch.tensor(A_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]


def train_graph_neural_network(
    model, train_loader, val_loader=None, 
    epochs=100, 
    learning_rate=0.001, 
    weight_decay=1e-5, 
    patience=10, 
    save_path="gnn_model.pth"
):
    """
    Training function with early stopping and validation
    """
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_true = []
        
        for X_batch, A_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(X_batch, A_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(outputs).detach().numpy())
            train_true.extend(y_batch.numpy())
        
        # Validation phase
        if val_loader:
            model.eval()
            val_losses = []
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for X_batch, A_batch, y_batch in val_loader:
                    outputs = model(X_batch, A_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    
                    val_preds.extend(torch.sigmoid(outputs).numpy())
                    val_true.extend(y_batch.numpy())
            
            # Compute metrics
            val_loss = np.mean(val_losses)
            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            
            # Logging and early stopping
            print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # # Early stopping
            # if patience_counter >= patience:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     break
    
    return model


def evaluate_graph_neural_network(model, dataloader):
    """
    Comprehensive model evaluation
    """
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for X_batch, A_batch, y_batch in dataloader:
            outputs = model(X_batch, A_batch)
            preds = torch.sigmoid(outputs).numpy().squeeze()
            preds_binary = (preds > 0.5).astype(int)
            
            all_preds.extend(preds_binary)
            all_true.extend(y_batch.numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds)
    recall = recall_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# (Previous code remains the same)

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
        'hidden_channels': 128,
        'num_gcn_layers': 4,
        'dnn_hidden_nodes': 512,
        'num_dnn_layers': 2,
        'dropout_rate': 0.33356257977269954,
        'l2_lambda': 0.0007517360053320633
    }

    # Prepare datasets
    train_dataset = GraphDataset(X_train, A_train, y_train)
    test_dataset = GraphDataset(X_test, A_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # train model and save weights
    model = GraphNeuralNetwork(**gnn_params)
    model = train_graph_neural_network(model, train_loader, val_loader=None, epochs=50, learning_rate=0.0009448, weight_decay=1e-5, patience=100, save_path="DeepHIT/weights/best_gnn_2.pth")


    # Initialize model
    # model = GraphNeuralNetwork(**gnn_params)
    # model.load_state_dict(torch.load('DeepHIT/weights/best_gnn.pth'))

    # # Evaluate the model
    evaluate_graph_neural_network(model, test_loader)
