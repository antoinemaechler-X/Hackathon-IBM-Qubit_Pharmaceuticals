# models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb


def create_model(model_name="random_forest", random_state=42, **kwargs):
    """
    Retourne un modèle scikit-learn en fonction du nom passé en paramètre.
    Vous pouvez ajouter ici tous les modèles souhaités.
    
    Paramètres
    ----------
    - model_name : str
        Nom du modèle à instancier. Ex: 'random_forest', 'logistic_regression', 'svc', etc.
    - random_state : int
        Graine pour la reproductibilité (utilisée par certains modèles).
    - kwargs : dict
        Paramètres supplémentaires à passer au constructeur du modèle.
    
    Retourne
    --------
    - Un estimateur scikit-learn (ex: RandomForestClassifier, LogisticRegression, etc.)
    """
    model_name = model_name.lower()

    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **kwargs)
    elif model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **kwargs)
    elif model_name == "svc":
        return SVC(random_state=random_state, **kwargs)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Modèle '{model_name}' inconnu. Choisissez parmi "
                         f"'random_forest', 'logistic_regression', 'svc', etc.")
