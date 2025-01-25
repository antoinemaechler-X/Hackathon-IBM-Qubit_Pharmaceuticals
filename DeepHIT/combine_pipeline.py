import models
import utils
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class ToxicityPipeline:
    def __init__(self):
        self.descriptor_model = None
        self.fingerprint_model = None
        self.graph_model = None
        self.pca = None

    def prepare_descriptor_data(self, X_raw):
        """
        Prepare the descriptor-based DNN dataset.
        """
        X, _ = utils.extract_selected_features(X_raw)
        return X.to_numpy()

    def prepare_fingerprint_data(self, X_raw):
        """
        Prepare the fingerprint-based DNN dataset with PCA.
        """
        ecfc_columns = [col for col in X_raw.columns if col.startswith('ecfc')]
        X = X_raw[ecfc_columns].to_numpy()
        if not self.pca:
            self.pca = PCA(n_components=30)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        return X_pca

    def prepare_graph_data(self, smiles_list):
        """
        Prepare the graph data for the GCN.
        """
        X, A, _ = utils.convert_to_graph(smiles_list)
        return X, A

    def load_models(self):
        """
        Load all trained models and set them to evaluation mode.
        """
        # Descriptor-based DNN
        self.descriptor_model = models.DescriptorBasedDNN(
            input_size=892, hidden_nodes=892, hidden_layers=4, dropout_rate=0.3
        )
        self.descriptor_model.load_state_dict(torch.load("DeepHIT/weights/descriptor_based_dnn.pth"))
        self.descriptor_model.eval()

        # Fingerprint-based DNN
        self.fingerprint_model = models.DescriptorBasedDNN(
            input_size=30, hidden_nodes=1024, hidden_layers=3, dropout_rate=0.5
        )
        self.fingerprint_model.load_state_dict(torch.load("DeepHIT/weights/fingerprint_dnn.pth"))
        self.fingerprint_model.eval()

        # Graph-based GCN
        self.graph_model = models.GCN_Model(
            in_features=65,
            gcn_hidden_nodes=64,
            gcn_hidden_layers=3,
            dnn_hidden_nodes=1024,
            dnn_hidden_layers=2,
            dropout_rate=0.1
        )
        self.graph_model.load_state_dict(torch.load("DeepHIT/weights/gcn.pth"))
        self.graph_model.eval()

    def predict(self, X_d, X_f, X_g, A_g):
        """
        Perform predictions using all models and combine the results.
        """
        with torch.no_grad():
            # Descriptor model predictions
            y_pred_d = self.descriptor_model(torch.tensor(X_d, dtype=torch.float32)).squeeze().round().numpy()

            # Fingerprint model predictions
            y_pred_f = self.fingerprint_model(torch.tensor(X_f, dtype=torch.float32)).squeeze().round().numpy()

            # Graph model predictions
            y_pred_g = self.graph_model(
                torch.tensor(X_g, dtype=torch.float32),
                torch.tensor(A_g, dtype=torch.float32)
            ).squeeze().round().numpy()

            # Combine predictions: return 1 if any model predicts 1
            y_pred = (y_pred_d + y_pred_f + y_pred_g) > 1
            return y_pred.astype(int)


if __name__ == '__main__':
    # Load data and train/test indices
    data = pd.read_csv('data/train.csv')
    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train_raw, X_test_raw = data.iloc[train_indices], data.iloc[test_indices]
    y_train, y_test = data['class'].iloc[train_indices], data['class'].iloc[test_indices]

    # Initialize pipeline
    pipeline = ToxicityPipeline()
    pipeline.load_models()

    # Prepare data
    X_d_train = pipeline.prepare_descriptor_data(X_train_raw)
    X_d_test = pipeline.prepare_descriptor_data(X_test_raw)
    X_f_train = pipeline.prepare_fingerprint_data(X_train_raw)
    X_f_test = pipeline.prepare_fingerprint_data(X_test_raw)
    X_g_train, A_train = pipeline.prepare_graph_data(X_train_raw['smiles'].tolist())
    X_g_test, A_test = pipeline.prepare_graph_data(X_test_raw['smiles'].tolist())

    # Predict
    y_pred = pipeline.predict(X_d_test, X_f_test, X_g_test, A_test)

    # Evaluate accuracy
    accuracy = np.mean(y_pred == y_test.to_numpy())
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Save the pipeline
    with open("DeepHIT/weights/toxicity_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("Pipeline saved to DeepHIT/weights/toxicity_pipeline.pkl")

    # Example: Load the pipeline and reuse it
    with open("DeepHIT/weights/toxicity_pipeline.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)
    print("Pipeline loaded successfully.")
import models
import utils
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class ToxicityPipeline:
    def __init__(self):
        self.descriptor_model = None
        self.fingerprint_model = None
        self.graph_model = None
        self.pca = None

    def prepare_descriptor_data(self, X_raw):
        """
        Prepare the descriptor-based DNN dataset.
        """
        X, _ = utils.extract_selected_features(X_raw)
        return X.to_numpy()

    def prepare_fingerprint_data(self, X_raw):
        """
        Prepare the fingerprint-based DNN dataset with PCA.
        """
        ecfc_columns = [col for col in X_raw.columns if col.startswith('ecfc')]
        X = X_raw[ecfc_columns].to_numpy()
        if not self.pca:
            self.pca = PCA(n_components=30)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        return X_pca

    def prepare_graph_data(self, smiles_list):
        """
        Prepare the graph data for the GCN.
        """
        X, A, _ = utils.convert_to_graph(smiles_list)
        return X, A

    def load_models(self):
        """
        Load all trained models and set them to evaluation mode.
        """
        # Descriptor-based DNN
        self.descriptor_model = models.DescriptorBasedDNN(
            input_size=892, hidden_nodes=892, hidden_layers=4, dropout_rate=0.3
        )
        self.descriptor_model.load_state_dict(torch.load("DeepHIT/weights/descriptor_based_dnn.pth"))
        self.descriptor_model.eval()

        # Fingerprint-based DNN
        self.fingerprint_model = models.DescriptorBasedDNN(
            input_size=30, hidden_nodes=1024, hidden_layers=3, dropout_rate=0.5
        )
        self.fingerprint_model.load_state_dict(torch.load("DeepHIT/weights/fingerprint_dnn.pth"))
        self.fingerprint_model.eval()

        # Graph-based GCN
        self.graph_model = models.GCN_Model(
            in_features=65,
            gcn_hidden_nodes=64,
            gcn_hidden_layers=3,
            dnn_hidden_nodes=1024,
            dnn_hidden_layers=2,
            dropout_rate=0.1
        )
        self.graph_model.load_state_dict(torch.load("DeepHIT/weights/gcn.pth"))
        self.graph_model.eval()

    def predict(self, X_d, X_f, X_g, A_g):
        """
        Perform predictions using all models and combine the results.
        """
        with torch.no_grad():
            # Descriptor model predictions
            y_pred_d = self.descriptor_model(torch.tensor(X_d, dtype=torch.float32)).squeeze().round().numpy()

            # Fingerprint model predictions
            y_pred_f = self.fingerprint_model(torch.tensor(X_f, dtype=torch.float32)).squeeze().round().numpy()

            # Graph model predictions
            y_pred_g = self.graph_model(
                torch.tensor(X_g, dtype=torch.float32),
                torch.tensor(A_g, dtype=torch.float32)
            ).squeeze().round().numpy()

            # Combine predictions: return 1 if any model predicts 1
            y_pred = (y_pred_d + y_pred_f + y_pred_g) > 1
            return y_pred.astype(int)


if __name__ == '__main__':
    # Load data and train/test indices
    data = pd.read_csv('data/train.csv')
    train_indices = np.load("DeepHIT/weights/train_indices.npy")
    test_indices = np.load("DeepHIT/weights/test_indices.npy")

    X_train_raw, X_test_raw = data.iloc[train_indices], data.iloc[test_indices]
    y_train, y_test = data['class'].iloc[train_indices], data['class'].iloc[test_indices]

    # Initialize pipeline
    pipeline = ToxicityPipeline()
    pipeline.load_models()

    # Prepare data
    X_d_train = pipeline.prepare_descriptor_data(X_train_raw)
    X_d_test = pipeline.prepare_descriptor_data(X_test_raw)
    X_f_train = pipeline.prepare_fingerprint_data(X_train_raw)
    X_f_test = pipeline.prepare_fingerprint_data(X_test_raw)
    X_g_train, A_train = pipeline.prepare_graph_data(X_train_raw['smiles'].tolist())
    X_g_test, A_test = pipeline.prepare_graph_data(X_test_raw['smiles'].tolist())

    # Predict
    y_pred = pipeline.predict(X_d_test, X_f_test, X_g_test, A_test)

    # Evaluate accuracy
    accuracy = np.mean(y_pred == y_test.to_numpy())
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Save the pipeline
    with open("DeepHIT/weights/toxicity_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("Pipeline saved to DeepHIT/weights/toxicity_pipeline.pkl")

    # Example: Load the pipeline and reuse it
    with open("DeepHIT/weights/toxicity_pipeline.pkl", "rb") as f:
        loaded_pipeline = pickle.load(f)
    print("Pipeline loaded successfully.")
