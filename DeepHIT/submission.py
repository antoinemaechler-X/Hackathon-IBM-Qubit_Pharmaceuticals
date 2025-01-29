import  models
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


def weighted_ensemble_proba(predictions, model_weights=None):
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
    return weighted_preds


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


def classify_new_data(new_csv_path, model_configs, train_data_path='data/train.csv'):
    """
    Classify a new CSV file, returning both classes and probabilities.
    
    Parameters:
    -----------
    new_csv_path : str
        Path to your new data CSV.
    model_configs : list
        List of model configurations (name, model, weights_path, input_type).
    train_data_path : str
        Path to the original training data (so we can mimic the transformations).
        
    Returns:
    --------
    pd.DataFrame
        A copy of the new data with additional columns:
        [ 'predicted_class', 'prediction_probability', 'ensemble_probability' ]
    """
    # 1. Load the original training data for consistent transformations
    train_data = pd.read_csv(train_data_path)

    # select the indices of the training data
    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    train_data = train_data.iloc[train_indices]
    
    # 2. Load the new data to classify
    new_data = pd.read_csv(new_csv_path)
    
    # 3. We do the same feature preparation on training + new data 
    #    so that we can use the same scalers / PCA, etc.
    
    #   A. Descriptor-based
    X_d_train, X_d_new = prepare_descriptor_data(train_data, new_data)
    
    #   B. Fingerprint-based
    X_f_train, X_f_new = prepare_fingerprint_data(train_data, new_data)
    
    #   C. Graph-based
    X_g_train, A_train, X_g_new, A_new = prepare_graph_data(train_data, new_data)
    
    # 4. Load each model, run inference
    #    We want probabilities -> apply sigmoid
    all_model_probs = []
    model_accuracies = []  
    with torch.no_grad():
        for config in model_configs:
            model = config['model']
            model.load_state_dict(torch.load(config['weights_path']))
            model.eval()
            
            # Identify which data to use
            if config['input_type'] == 'descriptor':
                logits = model(torch.tensor(X_d_new, dtype=torch.float32)).squeeze()
            elif config['input_type'] == 'fingerprint':
                logits = model(torch.tensor(X_f_new, dtype=torch.float32)).squeeze()
            else:  # graph
                logits = model(
                    torch.tensor(X_g_new, dtype=torch.float32),
                    torch.tensor(A_new, dtype=torch.float32)
                ).squeeze()
            
            # Convert logits to probabilities via sigmoid
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            all_model_probs.append(probabilities)

    model_weights = [0.8,0.74,0.8,0.9] 
    
    # 5. Compute the ensemble probability for each sample
    ensemble_probabilities = weighted_ensemble_proba(all_model_probs, model_weights)
    
    old_ensemble_probabilities = ensemble_probabilities

    if new_csv_path == "data/test_2.csv":
        # print repartition by series
        print("repartition by series : ", new_data['series'].value_counts(normalize=True))
        # predict by series
        for series in new_data['series'].unique():
            # get indices
            indices = new_data[new_data['series'] == series].index
            # get ensemble probabilities
            ensemble_probabilities = old_ensemble_probabilities[indices]
            
            # get the class
            predicted_classes = (ensemble_probabilities >= 0.5).astype(int)
            
            #print the repartition of the class
            print(f"repartition of the class : {series}", np.mean(predicted_classes))
            repartition = np.mean(predicted_classes)

            # rebalance the ensemble probabilities based on repartition
            proba = ensemble_probabilities*repartition
            denominateur = proba + (1-repartition)*(1-ensemble_probabilities)
            new_predicted_proba = proba/denominateur
            new_predicted_class= (new_predicted_proba >= 0.5).astype(int)
            #print the repartition of the class after rebalancing
            print(f"repartition of the class after rebalancing : {series}", np.mean(new_predicted_class))



        
        


    
    # 6. Threshold the ensemble probabilities to get final classification
    predicted_classes = (old_ensemble_probabilities >= 0.5).astype(int)

    # certainty = max(proba, 1-proba)
    ensemble_probabilities = np.maximum(old_ensemble_probabilities, 1 - old_ensemble_probabilities)

    
    # 7. Attach results to the new_data DataFrame
    new_data['ensemble_probability'] = ensemble_probabilities
    new_data['class'] = predicted_classes

    # sort data by ensemble probability
    new_data = new_data.sort_values(by='ensemble_probability', ascending=False)
    
    # keep smiles, ensemble_probability, predicted_class
    new_data = new_data[['smiles', 'class','ensemble_probability']]
    return new_data



if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/train.csv')

    data_1 = pd.read_csv('data/test_1.csv')

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
            'weights_path': "DeepHIT/weights/descriptor_based_dnn_2.pth",
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
            'weights_path': "DeepHIT/weights/fingerprint_dnn_2.pth",
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
            'weights_path': "DeepHIT/weights/gnn_2.pth",
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

    # Now run classification on a brand-new CSV, e.g. "data/new_data.csv"
    results_df = classify_new_data(
        new_csv_path="data/test_1.csv", 
        model_configs=model_configs, 
        train_data_path='data/train.csv'  # Your original training data
    )
    
    # Print or save the results
    print(results_df[['smiles', 'ensemble_probability', 'class']].head())
    results_df.to_csv("data/pred_1_2.csv", index=True)

    print(results_df['class'].value_counts(normalize=True))

    # same for test_2
    results_df = classify_new_data(
        new_csv_path="data/test_2.csv", 
        model_configs=model_configs, 
        train_data_path='data/train.csv'  # Your original training data
    )
    results_df.to_csv("data/pred_2_2.csv", index=True)

    # print proportions :
    print(results_df['class'].value_counts(normalize=True))
