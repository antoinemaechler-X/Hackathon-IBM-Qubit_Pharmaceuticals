import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def objective(trial):
    """
    Fonction d'objectif pour Optuna.
    Elle définit les hyperparamètres de XGBoost que l'on va faire varier,
    entraîne le modèle avec ces hyperparamètres et retourne une métrique
    (par ex. l'accuracy en validation croisée).
    """
    # Exemple de distribution d'hyperparamètres à explorer
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 12),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        # Utiliser GPU si vous le souhaitez, ex. 'tree_method': 'gpu_hist'
        'random_state': 42,  # pour la reproductibilité
        'eval_metric': 'logloss',
    }

    model = XGBClassifier(**params)

    # CV 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # On veut maximiser l'accuracy moyenne en CV
    return scores.mean()


if __name__ == "__main__":
    # =============================
    # 1. Ex. de dataset : Breast Cancer
    # =============================
    csv_path = "data/train.csv"
    data = pd.read_csv(csv_path)
    X = data.drop(columns=['class',"smiles"])
    y = data['class']

    # =============================
    # 2. Création de l'étude Optuna
    # =============================
    # direction='maximize' pour maximiser la metric
    study = optuna.create_study(direction='maximize')
    # Ou, si vous voulez un pruner (pour stopper rapidement les mauvais trials),
    # vous pouvez faire :
    # study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    # =============================
    # 3. Lancement de l'optimisation
    # =============================
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # =============================
    # 4. Résultats
    # =============================
    print("\n===== Résultats de l'optimisation =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value (accuracy): {study.best_trial.value:.4f}")
    print("  Params :")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # =============================
    # 5. Entraîner un modèle final
    #    avec les meilleurs hyperparamètres trouvés
    # =============================
    best_params = study.best_trial.params
    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X, y)

    # save the model
    import joblib
    joblib.dump(best_model, 'edouard/optuna/best_model_xgb.pkl')
    #save the study
    joblib.dump(study, 'edouard/optuna/study_xgb.pkl')

    # Exemple de prédiction : sur le même dataset (juste démonstration)
    # Bien sûr, en pratique, utilisez un set de test séparé
    y_pred = best_model.predict(X)
    acc_final = accuracy_score(y, y_pred)
    print(f"\nAccuracy sur l'ensemble : {acc_final:.4f}")
