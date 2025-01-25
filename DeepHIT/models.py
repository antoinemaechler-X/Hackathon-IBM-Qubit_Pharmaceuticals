import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DescriptorBasedDNN(nn.Module):
    def __init__(self, input_size, hidden_nodes, hidden_layers, dropout_rate):
        super(DescriptorBasedDNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.dropout_rate = dropout_rate
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_nodes, hidden_nodes) for _ in range(hidden_layers - 1)
        ])
        self.fc_out = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        for layer in self.fc_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.fc_out(x))
        return x


def run_descriptor_based_DNN(smiles_list,C_total, trained_model):
    # Convert data to PyTorch tensors
    C_total_tensor = torch.tensor(C_total, dtype=torch.float32)

    # Model initialization
    input_size = C_total.shape[1]  # number of features
    hidden_nodes = 892
    hidden_layers = 4
    dropout_rate = 0.2
    
    model = DescriptorBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
    
    # Load the trained model (assuming it is saved in .pt format)
    model.load_state_dict(torch.load(trained_model))
    model.eval()

    results = {}
    
    with torch.no_grad():
        # Make predictions
        y_pred = model(C_total_tensor).squeeze()

        for i, smi in enumerate(smiles_list):
            prob = y_pred[i].item()  # Convert tensor to native Python float
            results[smi] = prob

    return results

class FingerprintBasedDNN(nn.Module):
    def __init__(self, input_size, hidden_nodes, hidden_layers, dropout_rate):
        super(FingerprintBasedDNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.dropout_rate = dropout_rate
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_nodes, hidden_nodes) for _ in range(hidden_layers - 1)
        ])
        self.fc_out = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        for layer in self.fc_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.fc_out(x))
        return x
    
def run_fingerprint_based_DNN(smiles_list, X_total, trained_model):
    # Convert data to PyTorch tensors
    X_total_tensor = torch.tensor(X_total, dtype=torch.float32)

    # Model initialization
    input_size = X_total.shape[1]  # number of features
    hidden_nodes = 892
    hidden_layers = 4
    dropout_rate = 0.2
    
    model = FingerprintBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
    
    # Load the trained model (assuming it is saved in .pt format)
    model.load_state_dict(torch.load(trained_model))
    model.eval()

    results = {}
    
    with torch.no_grad():
        # Make predictions
        y_pred = model(X_total_tensor).squeeze()

        for i, smi in enumerate(smiles_list):
            prob = y_pred[i].item()  # Convert tensor to native Python float
            results[smi] = prob

    return results



class GraphConvLayer(nn.Module):
    """
    A simple graph convolution layer:
    1) Multiply adjacency matrix A by feature matrix X [batch_size, num_nodes, in_features]
    2) Apply learnable linear transform (in_features -> out_features)
    3) Reshape back to [batch_size, num_nodes, out_features]
    4) ReLU
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, X, A):
        """
        X: [batch_size, num_nodes, in_features]
        A: [batch_size, num_nodes, num_nodes]
        returns: [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, in_feats = X.shape
        
        # 1) Multiply adjacency matrix (per batch)
        out = torch.matmul(A, X)  # => [batch_size, num_nodes, in_feats]
        
        # 2) Flatten for linear transform => [batch_size * num_nodes, in_feats]
        out = out.view(batch_size * num_nodes, in_feats)
        
        # 3) Apply learnable linear transform => [batch_size * num_nodes, out_features]
        out = self.linear(out)
        
        # 4) Reshape back => [batch_size, num_nodes, out_features]
        out = out.view(batch_size, num_nodes, -1)
        
        # 5) ReLU activation
        out = F.relu(out)
        return out
    
# Define the Readout Layer
class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReadoutLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X):
        # X => [batch_size, num_nodes, hidden_dim]
        out = torch.sum(X, dim=1)         # => [batch_size, hidden_dim]
        out = F.relu(out)
        out = self.linear(out)            # => [batch_size, out_dim]
        return out
    

# Define the Model
class GCN_Model(nn.Module):
    def __init__(self, in_features, gcn_hidden_nodes, gcn_hidden_layers, dnn_hidden_nodes, dnn_hidden_layers,dropout_rate):
        super(GCN_Model, self).__init__()
        
        # GraphConvLayers
        self.gcn_layers = nn.ModuleList()
        # first GCN layer: in_features -> gcn_hidden_nodes
        self.gcn_layers.append(GraphConvLayer(in_features, gcn_hidden_nodes))
        # subsequent GCN layers: gcn_hidden_nodes -> gcn_hidden_nodes
        for _ in range(gcn_hidden_layers - 1):
            self.gcn_layers.append(GraphConvLayer(gcn_hidden_nodes, gcn_hidden_nodes))
        self.dropout = nn.Dropout(p=dropout_rate)

         # Readout layer
        self.readout = ReadoutLayer(gcn_hidden_nodes, dnn_hidden_nodes)  # <== define once

        # DNN layers
        self.dnn_layers = nn.ModuleList()
        # first DNN layer
        self.dnn_layers.append(nn.Linear(dnn_hidden_nodes, dnn_hidden_nodes))
        # self.dnn_layers.append(nn.Linear(gcn_hidden_nodes, dnn_hidden_nodes))
        # subsequent DNN layers
        for _ in range(dnn_hidden_layers - 1):
            self.dnn_layers.append(nn.Linear(dnn_hidden_nodes, dnn_hidden_nodes))
        
        # Output layer
        self.output_layer = nn.Linear(dnn_hidden_nodes, 1)

    def forward(self, X, A):
        # X => [batch_size, num_nodes, in_features]
        for layer in self.gcn_layers:
            X = layer(X, A)  # pass X, A => shape becomes [batch_size, num_nodes, gcn_hidden_nodes]
            X = self.dropout(X)
        
        
        # e.g. readout
        graph_feature = self.readout(X)
        
        # pass through DNN
        for layer in self.dnn_layers:
            graph_feature = torch.relu(layer(graph_feature))
        
        return self.output_layer(graph_feature)


# Training function
def run_graph_based_GCN(smiles_list, X_total, A_total, trained_model):
    # Hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.4
    num_nodes = X_total.shape[1]
    num_features = X_total.shape[2]
    dnn_hidden_nodes = 1024
    gcn_hidden_nodes = 64
    dnn_hidden_layers = 2
    gcn_hidden_layers = 3

    # Initialize the model
    model = GCN_Model(num_nodes, num_features, dnn_hidden_nodes, gcn_hidden_nodes, dnn_hidden_layers, gcn_hidden_layers)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Convert inputs to tensors
    X_tensor = torch.tensor(X_total, dtype=torch.float32)
    A_tensor = torch.tensor(A_total, dtype=torch.float32)

    # Load the trained model if available (for transfer learning or prediction)
    if trained_model is not None:
        model.load_state_dict(torch.load(trained_model))
    
    # Forward pass: Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor, A_tensor)
    
    # Convert predictions to probabilities
    y_predictions = y_pred.squeeze().tolist()

    # Map predictions to smiles
    results = {smiles_list[i]: y_predictions[i] for i in range(len(smiles_list))}

    return results