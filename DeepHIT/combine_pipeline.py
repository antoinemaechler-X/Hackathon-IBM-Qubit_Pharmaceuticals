import models
import utils
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import gnn
import GNN3


def prepare_descriptor_data(X_train_raw, X_test_raw):
    """
    Prepare the descriptor-based DNN dataset with more robust feature extraction.
    """
    X_train, feature_names = utils.extract_selected_features(X_train_raw)
    X_test, _ = utils.extract_selected_features(X_test_raw)
    
    # Optional: Add feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # X_train_scaled = X_train.to_numpy()
    # X_test_scaled = X_test.to_numpy()
    
    return X_train_scaled, X_test_scaled


def prepare_fingerprint_data(X_train_raw, X_test_raw):
    """
    Prepare the fingerprint-based DNN dataset with advanced PCA and scaling.
    """
    ecfc_columns = [col for col in X_train_raw.columns if col.startswith('ecfc')]
    X_train = X_train_raw[ecfc_columns].to_numpy()
    X_test = X_test_raw[ecfc_columns].to_numpy()
    
    # Combine PCA with scaling
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=30, random_state=42))
    ])
    
    X_train_processed = pca_pipeline.fit_transform(X_train)
    X_test_processed = pca_pipeline.transform(X_test)
    
    return X_train_processed, X_test_processed


def prepare_graph_data(X_train_raw, X_test_raw):
    """
    Prepare the graph data for the GCN with enhanced graph conversion.
    """
    X_train, A_train, node_features_train = utils.convert_to_graph(X_train_raw['smiles'].tolist())
    X_test, A_test, node_features_test = utils.convert_to_graph(X_test_raw['smiles'].tolist())
    return X_train, A_train, X_test, A_test


def weighted_ensemble_prediction(predictions, model_weights=None):
    """
    Perform weighted ensemble prediction.
    
    Args:
        predictions (list): List of model predictions
        model_weights (list, optional): Weights for each model's prediction
    
    Returns:
        numpy.ndarray: Final ensemble prediction
    """
    if model_weights is None:
        # Default to equal weights if not specified
        model_weights = [1/len(predictions)] * len(predictions)
    
    weighted_preds = np.average(predictions, axis=0, weights=model_weights)
    return (weighted_preds >= 0.5).astype(int)


def train_and_calibrate_models(X_train, y_train):
    """
    Train and calibrate models to improve prediction reliability.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
    
    Returns:
        List of calibrated models
    """
    # Placeholder for model training logic
    # You would replace this with actual model training code
    calibrated_models = []
    return calibrated_models


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/train.csv')

    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train_raw, X_test_raw = data.iloc[train_indices], data.iloc[test_indices]
    y_train, y_test = data['class'].iloc[train_indices], data['class'].iloc[test_indices]

    # Prepare data for different model types
    X_d_train, X_d_test = prepare_descriptor_data(X_train_raw, X_test_raw)
    X_f_train, X_f_test = prepare_fingerprint_data(X_train_raw, X_test_raw)
    X_g_train, A_train, X_g_test, A_test = prepare_graph_data(X_train_raw, X_test_raw)
    
    # Model configurations with performance tracking
    model_configs = [
        {
            'name': 'Descriptor DNN',
            'model': models.DescriptorBasedDNN(
                input_size=X_d_train.shape[1], 
                hidden_nodes=892, 
                hidden_layers=4, 
                dropout_rate=0.3
            ),
            'weights_path': "DeepHIT/weights/descriptor_based_dnn.pth",
            'input_type': 'descriptor'
        },
        {
            'name': 'Fingerprint DNN',
            'model': models.FingerprintBasedDNN(
                input_size=X_f_train.shape[1], 
                hidden_nodes=1024, 
                hidden_layers=3, 
                dropout_rate=0.5
            ),
            'weights_path': "DeepHIT/weights/fingerprint_dnn.pth",
            'input_type': 'fingerprint'
        },
        {
            'name': 'Graph Neural Network',
            'model': gnn.GraphNeuralNetwork(
                num_features=X_g_train.shape[2],
                hidden_channels=64,
                num_gcn_layers=3,
                dnn_hidden_nodes=256,
                num_dnn_layers=2,
                dropout_rate=0.2
            ),
            'weights_path': "DeepHIT/weights/gnn.pth",
            'input_type': 'graph'
        },
        {
            'name': 'Graph Neural Network (Updated)',
        #     'num_features': num_features,
        # 'hidden_channels': 128,
        # 'num_gcn_layers': 4,
        # 'dnn_hidden_nodes': 256,
        # 'num_dnn_layers': 2,
        # 'dropout_rate': 0.33356257977269954,
        # 'l2_lambda': 0.0007517360053320633
          'model': GNN3.GraphNeuralNetwork(
                num_features=X_g_train.shape[2],
                hidden_channels=128,
                num_gcn_layers=4,
                dnn_hidden_nodes=512,
                num_dnn_layers=2,
                dropout_rate=0.33356257977269954,
                l2_lambda=0.0007517360053320633
            ),
            'weights_path': "DeepHIT/weights/best_gnn.pth",
            'input_type': 'graph'
        }
    ]

    # Predict and track model performance
    model_predictions = []
    model_accuracies = []

    with torch.no_grad():
        for config in model_configs:
            model = config['model']
            model.load_state_dict(torch.load(config['weights_path']))
            model.eval()

            if config['input_type'] == 'descriptor':
                pred = model(torch.tensor(X_d_test, dtype=torch.float32)).squeeze()
            elif config['input_type'] == 'fingerprint':
                pred = model(torch.tensor(X_f_test, dtype=torch.float32)).squeeze()
                
            else:  # graph
                pred = model(
                    torch.tensor(X_g_test, dtype=torch.float32),
                    torch.tensor(A_test, dtype=torch.float32)
                )
                pred = torch.sigmoid(pred).squeeze()
                

            pred_np = pred.round().numpy()
            accuracy = np.mean(pred_np == y_test.to_numpy())
            
            model_predictions.append(pred_np)
            model_accuracies.append(accuracy)
            print(f"{config['name']} Accuracy: {accuracy:.4f}")

    # Compute model weights based on individual accuracies
    total_accuracy = sum(model_accuracies)
    model_weights = [acc/total_accuracy for acc in model_accuracies]

    # Weighted ensemble prediction
    y_pred = weighted_ensemble_prediction(model_predictions, model_weights)

    # Comprehensive evaluation
    accuracy = np.mean(y_pred == y_test.to_numpy())
    print(f"\nEnsemble Weighted Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))