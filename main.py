# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.utils.validation import check_is_fitted


from models import create_model  # Assurez-vous que ce module existe et que 'create_model' est bien défini

def load_and_prepare_data(csv_path, target_col, drop_cols=None, test_size=0.2, random_state=42):
    """
    Charger et préparer les données (X et y), avec séparation en jeu d'entraînement et test.
    """
    df = pd.read_csv(csv_path)
    if drop_cols is not None:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_and_save_model(model_name, model_params, X_train, y_train, do_scaling=True, save_path="mon_modele_preentraine.pkl"):
    """
    Crée un pipeline, entraîne un modèle, et le sauvegarde dans un fichier pkl.
    """
    steps = []
    if do_scaling:
        steps.append(('scaler', StandardScaler()))
    
    # Créer le modèle vierge à partir de create_model
    model = create_model(
        model_name=model_name,
        random_state=42,
        **model_params
    )
    steps.append(('classifier', model))
    
    pipeline = Pipeline(steps)
    
    # Entraîner le pipeline
    pipeline.fit(X_train, y_train)
    
    # Sauvegarder le modèle pré-entraîné
    joblib.dump(pipeline, save_path)
    print(f"Modèle enregistré sous '{save_path}'.")

def load_pretrained_model(path):
    """
    Charger un modèle pré-entraîné depuis un fichier pkl.
    """
    return joblib.load(path)

def evaluate_model(pipeline, X_test, y_test):
    """
    Évaluer un modèle pré-entraîné avec des données de test et afficher les métriques.
    """
    # check_is_fitted(pipeline)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.show()

if __name__ == "__main__":
    # Paramètres de données et modèle
    CSV_PATH = "data/train.csv"  # Remplacez par le chemin de votre fichier CSV
    TARGET_COL = "class"  # Remplacez par le nom de votre colonne cible
    DROP_COLS = ["smiles"]  # Liste des colonnes à supprimer (s'il y en a)
    
    # Charger et préparer les données
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        CSV_PATH,
        target_col=TARGET_COL,
        drop_cols=DROP_COLS
    )
    
    # 1) Entraîner et sauvegarder le modèle
    model_name = "xgboost"  # Choisir le modèle
    model_params = {"n_estimators": 100}  # Paramètres pour le modèle
    train_and_save_model(model_name, model_params, X_train, y_train, save_path="mon_xgb_pipeline.pkl")
    
    # 2) Charger le modèle pré-entraîné et l'évaluer
    pretrained_pipeline = load_pretrained_model("mon_xgb_pipeline.pkl")
    evaluate_model(pretrained_pipeline, X_test, y_test)
