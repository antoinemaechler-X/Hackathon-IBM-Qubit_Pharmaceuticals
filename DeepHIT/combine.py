import models
import utils
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def prepare_descriptor_data(X_train_raw, X_test_raw):
    """
    Prepare the descriptor-based DNN dataset.
    """
    X_train, _ = utils.extract_selected_features(X_train_raw)
    X_test, _ = utils.extract_selected_features(X_test_raw)
    return X_train.to_numpy(), X_test.to_numpy()


def prepare_fingerprint_data(X_train_raw, X_test_raw):
    """
    Prepare the fingerprint-based DNN dataset with PCA.
    """
    ecfc_columns = [col for col in X_train_raw.columns if col.startswith('ecfc')]
    X_train = X_train_raw[ecfc_columns].to_numpy()
    X_test = X_test_raw[ecfc_columns].to_numpy()
    
    pca = PCA(n_components=30)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def prepare_graph_data(X_train_raw, X_test_raw):
    """
    Prepare the graph data for the GCN.
    """
    X_train, A_train, _ = utils.convert_to_graph(X_train_raw['smiles'].tolist())
    X_test, A_test, _ = utils.convert_to_graph(X_test_raw['smiles'].tolist())
    return X_train, A_train, X_test, A_test


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/train.csv')

    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train_raw, X_test_raw = data.iloc[train_indices], data.iloc[test_indices]
    y_train, y_test = data['class'].iloc[train_indices], data['class'].iloc[test_indices]

    
    # Descriptor-based DNN
    X_d_train, X_d_test = prepare_descriptor_data(X_train_raw, X_test_raw)
    descriptor_model = models.DescriptorBasedDNN(
        input_size=X_d_train.shape[1], hidden_nodes=892, hidden_layers=4, dropout_rate=0.3
    )
    descriptor_model.load_state_dict(torch.load("DeepHIT/weights/descriptor_based_dnn.pth"))
    descriptor_model.eval()

    # Fingerprint-based DNN
    X_f_train, X_f_test = prepare_fingerprint_data(X_train_raw, X_test_raw)
    fingerprint_model = models.DescriptorBasedDNN(
        input_size=X_f_train.shape[1], hidden_nodes=1024, hidden_layers=3, dropout_rate=0.5
    )
    fingerprint_model.load_state_dict(torch.load("DeepHIT/weights/fingerprint_dnn.pth"))
    fingerprint_model.eval()

    # Graph-based GCN
    X_g_train, A_train, X_g_test, A_test = prepare_graph_data(X_train_raw, X_test_raw)
    graph_model = models.GCN_Model(
        in_features=X_g_train.shape[2],
        gcn_hidden_nodes=64,
        gcn_hidden_layers=3,
        dnn_hidden_nodes=1024,
        dnn_hidden_layers=2,
        dropout_rate=0.1
    )
    graph_model.load_state_dict(torch.load("DeepHIT/weights/gcn.pth"))
    graph_model.eval()

    # Predict
    with torch.no_grad():
        y_pred_d = descriptor_model(torch.tensor(X_d_test, dtype=torch.float32)).squeeze().round().numpy()
        y_pred_f = fingerprint_model(torch.tensor(X_f_test, dtype=torch.float32)).squeeze().round().numpy()
        y_pred_g = graph_model(
            torch.tensor(X_g_test, dtype=torch.float32),
            torch.tensor(A_test, dtype=torch.float32)
        ).squeeze().round().numpy()

        # Combine predictions: return 1 if any model predicts 1
        y_pred = (y_pred_d + y_pred_f + y_pred_g) > 0
        y_pred = y_pred.astype(int)

        # Evaluate accuracy
        accuracy = np.mean(y_pred == y_test.to_numpy())
        print(f"Accuracy on test set: {accuracy:.4f}")
