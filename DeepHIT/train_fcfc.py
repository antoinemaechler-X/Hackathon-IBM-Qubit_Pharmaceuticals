import models
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define Dataset Class
class FingerprintDataset(Dataset):
    """
    Dataset for handling fingerprint-based features and labels for PyTorch DataLoader.
    """
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # Reshape to match (batch_size, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Training Function
def train_dnn_with_dataloader(X_train, y_train, model, batch_size=32, learning_rate=0.001, epochs=100, save_path="descriptor_based_dnn.pth",X_test=None,y_test=None):
    """
    Train a DNN model using a PyTorch DataLoader.
    """
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

    # DataLoader for batch processing
    dataset = FingerprintDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with progress bar
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            y_pred = model(inputs)


            # Compute loss
            loss = criterion(y_pred, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log epoch loss
        if epoch % 10 == 0:
            model.eval()
            avg_loss = running_loss / len(train_loader)
            train_acc = np.mean(predict_dnn(model, X_train) == y_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")
        # early stopping
        if X_test is not None and y_test is not None:
            y_pred = predict_dnn(model, X_test)
            val_acc = np.mean(y_pred == y_test)
            print(f"Validation Accuracy: {val_acc:.4f}")
            if train_acc > val_acc+0.07:
                break

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model trained and saved at {save_path}")


# Define Prediction Function
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
    
def preprocess_data(X_train, X_test, n_components=30):
    """
    Scale and apply PCA to the data consistently.
    """
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca,pca


# Main Execution
if __name__ == '__main__':
    # Load and preprocess the dataset
    data = pd.read_csv("data/train.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract fingerprint features (columns starting with 'fcfc') and target labels
    fcfc_columns = [col for col in data.columns if col.startswith('fcfc')]
    X = data[fcfc_columns].to_numpy()
    y = data['class'].to_numpy()

    # scale data
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

   

    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    # X_train = X
    # y_train = y
    # X_test = X
    # y_test = y
    X_train_pca, X_test_pca, pca = preprocess_data(X_train, X_test, n_components=30)

    # Define DNN model parameters
    input_size = X_train_pca.shape[1]
    hidden_nodes = 1024
    hidden_layers = 3
    dropout_rate = 0.5

    # Initialize the DNN model
    model = models.FingerprintBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
    model.to(device)

    # Train the model
    train_dnn_with_dataloader(X_train_pca, y_train, model, batch_size=32, learning_rate=0.001, epochs=100, save_path="DeepHIT/weights/fcfc.pth")

    # Load the trained model
    model.load_state_dict(torch.load("DeepHIT/weights/fcfc.pth"))
    model.eval()

    # Make predictions on the test set
    y_pred = predict_dnn(model, X_test_pca)

    # Evaluate model accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")
