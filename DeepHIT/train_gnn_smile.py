import models
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class DescriptorDataset(Dataset):
    def __init__(self, X_data,A, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # reshape y pour correspondre à la forme (batch_size, 1)
        
    def __len__(self):
        return len(self.X)  # Le nombre d'exemples dans le dataset
    
    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]

# Fonction de training avec DataLoader


def train_gcn(model, X_train, A_train, Y_train, epochs=100, learning_rate=0.01, batch_size=128, name="gcn.pth"):
    """
    Fonction d'entraînement pour le modèle Graph Neural Network (GNN).
    
    :param model: Le modèle GNN à entraîner.
    :param X_train: La matrice des caractéristiques des nœuds d'entraînement (dimensions: [num_nodes, num_features]).
    :param A_train: La matrice d'adjacence du graphe d'entraînement (dimensions: [num_nodes, num_nodes]).
    :param Y_train: Les labels d'entraînement (dimensions: [num_samples, 1]).
    :param epochs: Le nombre d'époques pour l'entraînement (par défaut: 100).
    :param learning_rate: Le taux d'apprentissage pour l'optimiseur (par défaut: 0.01).
    :param batch_size: La taille des lots (par défaut: 32).
    
    :return: Le modèle GNN entraîné.
    """
    # Mettre le modèle en mode entraînement
    model.train()
    
    # Définir l'optimiseur et la fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Utilisation d'une perte binaire pour classification binaire
    
    
    num_samples = X_train.shape[0]
    dataloader = DataLoader(DescriptorDataset(X_train, A_train, Y_train), batch_size=batch_size, shuffle=True)


    # Boucle d'entraînement sur les époques
    for epoch in range(epochs):

        # Initialiser la perte à 0
        loss = 0
        for i, (X_batch, A_batch, Y_batch) in enumerate(dataloader):
            # Remettre à zéro les gradients
            optimizer.zero_grad()
            
            # Prédiction du modèle
            Y_pred = model(X_batch, A_batch)
            
            # Calcul de la perte
            loss = criterion(Y_pred, Y_batch)
            
            # Rétropropagation
            loss.backward()
            
            # Mise à jour des poids
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), name)


    
    return model

if __name__ == '__main__':
    # Charger les données
    data = pd.read_csv("data/train.csv")
    data = data
    smiles_list = data["smiles"].tolist()
    print(data.shape)
    X,A,smiles_list = utils.convert_to_graph(smiles_list)
    y = data['class'].to_numpy()

    print(A)
    # print(y)

    print(f"X length: {len(X)}")
    print(f"A length: {len(A)}")
    print(f"y length: {len(y)}")
    

    #split the data
    X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(X, A, y, test_size=0.2, random_state=42)
    print(A_train.shape)
    print(X_train.shape)

    input_size = X_train.shape[2]  # Nombre de features
    num_nodes, num_features, dnn_hidden_nodes, gcn_hidden_nodes, dnn_hidden_layers, gcn_hidden_layers = X_train.shape[0], X_train.shape[2], 1024, 64, 2, 3
    

    # Initialiser le modèle DNN
    model = models.GCN_Model(num_features, gcn_hidden_nodes, gcn_hidden_layers, dnn_hidden_nodes, dnn_hidden_layers,dropout_rate=0.1)
    
    # Entraîner le modèle
    trained_model = train_gcn(model, X_train, A_train, y_train, learning_rate=0.0001, epochs=100, name="gcn.pth")

    # Charger le modèle entraîné
    model.load_state_dict(torch.load("gcn.pth"))
    model.eval()  # Passer le modèle en mode évaluation
    
    # Prédiction sur les données de test
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32), torch.tensor(A_test, dtype=torch.float32))
        y_pred = torch.sigmoid(y_pred).numpy().squeeze()
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy on test set: {accuracy:.4f}")
    # accuracy on train set
    with torch.no_grad():
        y_pred = model(torch.tensor(X_train, dtype=torch.float32), torch.tensor(A_train, dtype=torch.float32))
        y_pred = torch.sigmoid(y_pred).numpy().squeeze()
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_train)
        print(f"Accuracy on train set: {accuracy:.4f}")
