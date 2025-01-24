import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import display
from IPython.display import display

# 1. Chargement des données
# Remplacez 'data.csv' par le chemin réel de votre fichier
df = pd.read_csv('data/train.csv')

# 2. Exploration initiale
print("Dimensions du dataset :", df.shape)
print("\nAperçu du dataset :")
display(df.head())  # Utilisez simplement df.head() si vous n'êtes pas sur un notebook
print("\nInformations générales :")
df.info()
print("\nStatistiques descriptives :")
display(df.describe())

# 3. Vérification des valeurs manquantes (NA)
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

# 4. Suppression éventuelle de colonnes non pertinentes 
# (Exemple : la colonne 'smiles' pourrait être conservée comme identifiant 
# ou supprimée si elle n'est pas nécessaire pour la classification)
if 'smiles' in df.columns:
    df.drop(columns=['smiles'], inplace=True)

# 5. Définir la variable cible (target) et les features
# Remplacez 'target_column' par le nom réel de votre colonne cible.
target_col = 'class'  # À adapter
X = df.drop(columns=[target_col])  # Tout sauf la cible
y = df[target_col]

# 6. Séparer les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% pour le test (à ajuster)
    random_state=42,    # Pour la reproductibilité
    stratify=y          # Si votre target est catégorielle, c'est mieux de stratifier
)

# 7. Normalisation ou standardisation des features (optionnel mais souvent utile)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Entraînement d’un premier modèle simple (Random Forest par exemple)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Évaluation sur les données de test
y_pred = model.predict(X_test_scaled)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
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