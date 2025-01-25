import models
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train.csv')

# descriptor-based DNN
smiles_list = data['smiles'].tolist()
X_d, smiles_list = utils.extract_selected_features(data)
X_d = X_d.to_numpy()
y = data['class']
X_d_train, X_d_test, y_train, y_test = train_test_split(X_d, y, test_size=0.2, random_state=42)

input_size = X_d.shape[1]  # Nombre de features
hidden_nodes = 892  # Nombre de neurones dans chaque couche cachée
hidden_layers = 4 # Nombre de couches cachées
dropout_rate = 0.3  # Taux de dropout
modelDNN = models.DescriptorBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
modelDNN.load_state_dict(torch.load("DeepHIT/weights/descriptor_based_dnn.pth"))
modelDNN.eval()  # Passer le modèle en mode évaluation

# fingerprint-based DNN
ecfc_columns = [col for col in data.columns if col.startswith('ecfc')]
X_f = data[ecfc_columns]
y = data['class']
X_f = X_f.to_numpy()
X_f_train, X_f_test, y_train, y_test = train_test_split(X_f, y, test_size=0.2, random_state=42)
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_f)
X_pca_test = pca.transform(X_f_test)
input_size = X_pca.shape[1]  # Nombre de features
hidden_nodes = 1024  # Nombre de neurones dans chaque couche cachée
hidden_layers = 3 # Nombre de couches cachées
dropout_rate = 0.5  # Taux de dropout
modelFingerprint = models.DescriptorBasedDNN(input_size, hidden_nodes, hidden_layers, dropout_rate)
modelFingerprint.load_state_dict(torch.load("DeepHIT/weights/fingerprint_dnn.pth"))
modelFingerprint.eval()  # Passer le modèle en mode évaluation

# GCN
X, A, smiles_list = utils.convert_to_graph(smiles_list)
y = data['class'].to_numpy()
X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(X, A, y, test_size=0.2, random_state=42)
input_size = X.shape[2]  # Nombre de features
num_nodes, num_features, dnn_hidden_nodes, gcn_hidden_nodes, dnn_hidden_layers, gcn_hidden_layers = X.shape[0], X.shape[2], 1024, 64, 2, 3
modelGCN = models.GCN_Model(num_features, gcn_hidden_nodes, gcn_hidden_layers, dnn_hidden_nodes, dnn_hidden_layers,dropout_rate=0.1)
modelGCN.load_state_dict(torch.load("DeepHIT/weights/gcn.pth"))
modelGCN.eval()  # Passer le modèle en mode évaluation


# predict
with torch.no_grad():
    y_pred_d = modelDNN(torch.tensor(X_d_test, dtype=torch.float32)).squeeze().round().numpy()
    y_pred_f = modelFingerprint(torch.tensor(X_pca_test, dtype=torch.float32)).squeeze().round().numpy()
    y_pred_g = modelGCN(torch.tensor(X_test, dtype=torch.float32), torch.tensor(A_test, dtype=torch.float32)).squeeze().round().numpy()
    
    # if one is 1, the molecule is toxic
    y_pred = (y_pred_d + y_pred_f + y_pred_g) > 1
    y_pred = y_pred.astype(int)

    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")

