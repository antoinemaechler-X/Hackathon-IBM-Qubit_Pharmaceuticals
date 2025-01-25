import models
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class fingerprintDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # reshape y pour correspondre à la forme (batch_size, 1)
        
    def __len__(self):
        return len(self.X)  # Le nombre d'exemples dans le dataset
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Fonction de training avec DataLoader
def train_dnn_with_dataloader(X_train, y_train, model, batch_size=32, learning_rate=0.001, epochs=100, name="descriptor_based_dnn_with_dataloader.pth"):
    # Initialiser le modèle
    
    # Fonction de perte
    criterion = nn.BCELoss()
    
    # Optimiseur
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Créer le dataset et le DataLoader
    dataset = fingerprintDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Entraînement du modèle
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Passage avant
            y_pred = model(inputs)
            
            # Calcul de la perte
            loss = criterion(y_pred, labels)
            
            # Passage arrière
            optimizer.zero_grad()  # Réinitialiser les gradients
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids
            
            running_loss += loss.item()
        
        # Affichage de la perte moyenne après chaque époque
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
            num_samples = len(X_train)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            model.eval()
            y_pred = model(X_train_tensor).squeeze()
            y_pred = y_pred.round()
            y_pred = y_pred.detach().numpy()
            acc = (y_pred == y_train).sum().item() / num_samples
            print(f"Train Accuracy: {acc:.4f}")
    
    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), name)
    print(f"Model trained and saved as {name}")


def predict_dnn(model, X_test):
    """
    Effectue des prédictions sur de nouvelles données.
    
    :param model: Le modèle préalablement entraîné.
    :param X_test: Les nouvelles données pour lesquelles prédire les résultats.
    :return: Les prédictions effectuées par le modèle.
    """
    model.eval()  # Mettre le modèle en mode évaluation
    with torch.no_grad():  # Désactiver le calcul des gradients pour accélérer la prédiction
        # Convertir les données d'entrée en tenseur
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Effectuer les prédictions
        y_pred = model(X_test_tensor).squeeze()
        
        # Appliquer un seuil de 0.5 pour la classification binaire
        y_pred_class = y_pred.round() # Convertir en 0 ou 1
        y_pred_class = y_pred_class.numpy()
        
    return y_pred_class




if __name__ == '__main__':
    # Charger les données
    data = pd.read_csv("data/train.csv")
    X, smiles_list = utils.extract_selected_features(data)
    y = data['class']
    # X is a dataframe of features, put it in a numpy array
    X = X.to_numpy()

    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]  # Nombre de features
    hidden_nodes = 892  # Nombre de neurones dans chaque couche cachée
    hidden_layers = 4 # Nombre de couches cachées
    dropout_rate = 0.3  # Taux de dropout
    

    # Initialiser le modèle DNN
    model = models.DescriptorBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
    
    # Entraîner le modèle
    trained_model = train_dnn_with_dataloader(X_train, y_train, model, learning_rate=0.0001, epochs=100, name="descriptor_based_dnn.pth")

    # Charger le modèle entraîné
    model.load_state_dict(torch.load("descriptor_based_dnn.pth"))
    model.eval()  # Passer le modèle en mode évaluation
    
    # Prédiction  # Exemple de données de test
    y_pred = predict_dnn(model, X_test)
    # accuracy
    print(y_pred[0:10])
    print(y_test[0:5])
    accuracy = (y_pred == y_test).sum().item() / len(y_test)
    print(f"Accuracy: {accuracy:.4f}")
