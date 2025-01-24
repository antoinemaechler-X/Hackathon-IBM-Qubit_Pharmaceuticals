# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from models import create_model  # votre module qui crée un modèle vierge

def load_and_prepare_data(csv_path, target_col, drop_cols=None, test_size=0.2, random_state=42):
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
    # Créer un pipeline
    steps = []
    if do_scaling:
        steps.append(('scaler', StandardScaler()))
    
    # Modèle vierge
    model = create_model(
        model_name=model_name,
        random_state=42,
        **model_params
    )
    steps.append(('classifier', model))
    
    pipeline = Pipeline(steps)
    
    # Entraîner
    pipeline.fit(X_train, y_train)
    
    # Sauvegarder
    joblib.dump(pipeline, save_path)
    print(f"Modèle enregistré sous '{save_path}'.")

def load_pretrained_model(path):
    return joblib.load(path)

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion (modèle pré-entraîné)")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.show()

if __name__ == "__main__":
    # Paramètres
    CSV_PATH = "data/train.csv"
    TARGET_COL = "class"
    DROP_COLS = ["smiles"]
    
    # Charger les données
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        CSV_PATH,
        target_col=TARGET_COL,
        drop_cols=DROP_COLS
    )
    
    # 1) ENTRAINER ET SAUVEGARDER UN MODELE
    model_name = "xgboost"
    model_params = {"n_estimators": 100}
    train_and_save_model(model_name, model_params, X_train, y_train, save_path="mon_xgb_pipeline.pkl")
    
    # 2) CHARGER UN MODELE PRE-ENTRAINE ET L'EVALUER
    pretrained_pipeline = load_pretrained_model("mon_xgb_pipeline.pkl")
    evaluate_model(pretrained_pipeline, X_test, y_test)