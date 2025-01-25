import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import sklearn.metrics as metrics

import models
import utils


class AdaptiveBatchNorm1d(nn.Module):
    """
    Adaptive Batch Normalization that can handle varying input sizes
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        """
        Adaptive forward pass that works with different input shapes
        """
        if x.dim() == 2:
            # If input is 2D, treat it as (batch, features)
            return F.batch_norm(
                x, 
                self.running_mean[:x.size(1)], 
                self.running_var[:x.size(1)], 
                self.weight[:x.size(1)], 
                self.bias[:x.size(1)], 
                self.training, 
                self.momentum, 
                self.eps
            )
        elif x.dim() == 3:
            # If input is 3D, treat it as (batch, nodes, features)
            # Reshape to apply batch norm along the last dimension
            orig_shape = x.shape
            x_reshaped = x.view(-1, orig_shape[-1])
            
            # Apply batch norm
            x_normed = F.batch_norm(
                x_reshaped, 
                self.running_mean[:orig_shape[-1]], 
                self.running_var[:orig_shape[-1]], 
                self.weight[:orig_shape[-1]], 
                self.bias[:orig_shape[-1]], 
                self.training, 
                self.momentum, 
                self.eps
            )
            
            # Reshape back to original
            return x_normed.view(orig_shape)
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()}")


class GraphNeuralNetwork(nn.Module):
    def __init__(self, config):
        """
        Enhanced Graph Neural Network with adaptive batch normalization
        """
        super(GraphNeuralNetwork, self).__init__()
        
        # Regularization parameters
        self.dropout_rate = config.dropout_rate
        self.l2_lambda = config.l2_lambda
        
        # Graph Convolution Layers
        self.gcn_layers = nn.ModuleList()
        
        # Input layer
        self.gcn_layers.append(
            nn.Sequential(
                nn.Linear(config.num_features, config.hidden_channels),
                nn.ReLU(),
                AdaptiveBatchNorm1d(config.hidden_channels),
                nn.Dropout(self.dropout_rate)
            )
        )
        
        # Additional GCN layers
        for _ in range(config.num_gcn_layers - 1):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(config.hidden_channels, config.hidden_channels),
                    nn.ReLU(),
                    AdaptiveBatchNorm1d(config.hidden_channels),
                    nn.Dropout(self.dropout_rate)
                )
            )
        
        # Dense Layers
        self.dnn_layers = nn.ModuleList()
        input_size = config.hidden_channels
        
        for _ in range(config.num_dnn_layers):
            self.dnn_layers.append(
                nn.Sequential(
                    nn.Linear(input_size, config.dnn_hidden_nodes),
                    nn.ReLU(),
                    AdaptiveBatchNorm1d(config.dnn_hidden_nodes),
                    nn.Dropout(self.dropout_rate)
                )
            )
            input_size = config.dnn_hidden_nodes
        
        # Final output layer
        self.output_layer = nn.Linear(input_size, 1)
    
    def forward(self, X, A):
        """
        Forward pass with graph convolution and regularization.
        """
        # Graph convolution
        for layer in self.gcn_layers:
            X = self._graph_convolution(X, A, layer)
        
        # Global pooling
        X = torch.mean(X, dim=1)
        
        # Dense layers
        for layer in self.dnn_layers:
            X = self._dense_layer(X, layer)
        
        # Output layer
        return self.output_layer(X)
    
    def _graph_convolution(self, X, A, layer):
        """
        Enhanced graph convolution with adjacency normalization.
        """
        # Normalize adjacency matrix
        degrees = torch.sum(A, dim=2)
        D_inv_sqrt = torch.pow(degrees + 1e-7, -0.5)
        A_norm = A * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(-2)
        
        # Linear transformation
        X = layer[0](X)
        
        # Message passing
        X = torch.bmm(A_norm, X)
        
        # Activation and normalization
        X = layer[1](X)
        X = layer[2](X)
        
        # Dropout
        X = layer[3](X)
        
        return X
    
    def _dense_layer(self, X, layer):
        """
        Dense layer with regularization.
        """
        X = layer[0](X)
        X = layer[1](X)
        X = layer[2](X)
        X = layer[3](X)
        return X


class GraphDataset(Dataset):
    def __init__(self, X_data, A_data, y_data):
        """
        Dataset with data augmentation support.
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.A = torch.tensor(A_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]


def train_and_validate(config=None):
    """
    Training function with wandb integration and advanced techniques.
    """
    # Initialize wandb run
    with wandb.init(config=config):
        # Get the configuration
        config = wandb.config
        
        # Load data
        data = pd.read_csv("data/train.csv")
        smiles_list = data["smiles"].tolist()
        y = data['class'].to_numpy()

        # Convert to graph data
        X, A, smiles_list = utils.convert_to_graph(smiles_list)

        # Load train/test indices
        train_indices = np.load("DeepHIT/weights/train_indices.npy")
        val_indices = np.load("DeepHIT/weights/test_indices.npy")

        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        A_train, A_val = A[train_indices], A[val_indices]

        # Prepare datasets
        train_dataset = GraphDataset(X_train, A_train, y_train)
        val_dataset = GraphDataset(X_val, A_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # Model configuration
        config.num_features = X_train.shape[2]
        model = GraphNeuralNetwork(config)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.l2_lambda
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(config.epochs):
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
                
                # Collect predictions for train metrics
                preds = torch.sigmoid(outputs).detach().numpy()
                train_preds.extend(preds)
                train_true.extend(y_batch.numpy())
            
            # Validation phase
            model.eval()
            val_losses = []
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for X_batch, A_batch, y_batch in val_loader:
                    outputs = model(X_batch, A_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    
                    # Collect predictions
                    preds = torch.sigmoid(outputs).numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())
            
            # Compute metrics
            # Train metrics
            train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
            train_accuracy = np.mean(train_preds_binary == train_true)
            
            # Validation metrics
            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            val_accuracy = np.mean(val_preds_binary == val_true)
            
            precision = metrics.precision_score(val_true, val_preds_binary)
            recall = metrics.recall_score(val_true, val_preds_binary)
            f1 = metrics.f1_score(val_true, val_preds_binary)
            
            # Log metrics to wandb
            wandb.log({
                'train_loss': np.mean(train_losses),
                'val_loss': np.mean(val_losses),
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping
            current_val_loss = np.mean(val_losses)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(current_val_loss)
            
            # Early stopping condition
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Return best validation metrics
        return best_val_loss


def main():
    # Wandb configuration
    wandb.login()
    
    # Define sweep configuration
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 1e-3},
            'batch_size': {'values': [32, 64, 128]},
            'hidden_channels': {'values': [32, 64, 128]},
            'num_gcn_layers': {'values': [2, 3, 4]},
            'num_dnn_layers': {'values': [1, 2, 3]},
            'dnn_hidden_nodes': {'values': [128, 256, 512]},
            'dropout_rate': {'min': 0.1, 'max': 0.5},
            'l2_lambda': {'min': 1e-5, 'max': 1e-3},
            'epochs': {'value': 50}
        }
    }
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_configuration, project='graph-neural-network-optimization')
    
    # Run sweep
    wandb.agent(sweep_id, function=train_and_validate, count=50)


if __name__ == '__main__':
    main()