# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# On importe la fonction de création de modèles
from models import create_model  # <-- module que vous avez créé

def load_and_prepare_data(csv_path, target_col, drop_cols=None, test_size=0.2, random_state=42):
    # 1. Chargement
    df = pd.read_csv(csv_path)

    # 2. Suppression de colonnes inutiles
    if drop_cols is not None:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # 3. Séparation features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 4. Division en train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test, do_scaling=True):
    # Construction d'un pipeline simple (scaler + modèle), si nécessaire
    steps = []
    if do_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', model))
    
    pipeline = Pipeline(steps)
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    
    # Évaluation
    print("\n===== ÉVALUATION DU MODÈLE =====")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.show()

def evaluate(model,data):
    """
    display metrics for the model
    """
    y_pred = model.predict(data)
    print("\n===== ÉVALUATION DU MODÈLE =====")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.show()

if __name__ == "__main__":
    # Paramètres de base
    CSV_PATH = "data/train.csv"
    TARGET_COL = "class"   # Nom de la colonne cible
    DROP_COLS = ["smiles"]      # Exemple
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Chargement et préparation
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        CSV_PATH,
        target_col=TARGET_COL,
        drop_cols=DROP_COLS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # CHOIX DU MODÈLE : juste en changeant 'model_name', on change d'algo
    # ex: "random_forest", "logistic_regression", "svc" 
    model_name = "xgboost"  
    model_params = {
        "n_estimators": 100  # ex. paramètre spécifique à RandomForest
    }

    # Création du modèle à partir de models.py
    model = create_model(
        model_name=model_name, 
        random_state=RANDOM_STATE, 
        **model_params
    )
    
    # Entraînement et évaluation
    train_and_evaluate(model, X_train, X_test, y_train, y_test, do_scaling=True)
