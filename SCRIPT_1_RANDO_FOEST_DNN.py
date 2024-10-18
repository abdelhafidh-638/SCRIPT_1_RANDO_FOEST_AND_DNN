# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Chargement du dataset
dataset_path = r'C:\Users\win\Desktop\DB\UNSW_NB15_training-set.csv'
df = pd.read_csv(dataset_path)

# Sélection des caractéristiques dynamiquement
selected_features = list(df.columns[:-1])  # Exclut la dernière colonne qui est généralement la variable cible
target_variable = df.columns[-1]  # Nom de la variable cible

# Exclusion des colonnes non numériques
numeric_features = df[selected_features].select_dtypes(include=['number']).columns
selected_features = list(numeric_features)

# Exclusion de la variable cible pour X
X = df[selected_features]

# Normalisation
scaler = StandardScaler()
X.loc[:, selected_features] = scaler.fit_transform(X.loc[:, selected_features])

# Codage (exemple pour encoder les variables catégorielles, ajustez selon vos besoins)
# Exemple avec LabelEncoder pour toutes les colonnes catégorielles
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# Équilibrage des données avec SMOTE
y = df[target_variable]
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Réduction des caractéristiques avec SelectKBest (test ANOVA)
k_best_features = 10  # Remplacez par le nombre souhaité de caractéristiques à conserver
selector = SelectKBest(f_classif, k=k_best_features)
X_resampled_selected = selector.fit_transform(X_resampled, y_resampled)

# Répartition des données (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_resampled_selected, y_resampled, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Prédictions
rf_predictions = rf_model.predict(X_test)

# Évaluation du modèle Random Forest
print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("Recall:", recall_score(y_test, rf_predictions))
print("F1 Score:", f1_score(y_test, rf_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_predictions))

# Entraînement du modèle Réseaux de Neurones Profonds (DNN)
dnn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)  # Ajoutez des paramètres selon vos besoins
dnn_model.fit(X_train, y_train)

# Prédictions
dnn_predictions = dnn_model.predict(X_test)

# Évaluation du modèle Réseaux de Neurones Profonds (DNN)
print("\nDeep Neural Network Metrics:")
print("Accuracy:", accuracy_score(y_test, dnn_predictions))
print("Precision:", precision_score(y_test, dnn_predictions))
print("Recall:", recall_score(y_test, dnn_predictions))
print("F1 Score:", f1_score(y_test, dnn_predictions))
print("ROC-AUC Score:", roc_auc_score(y_test, dnn_predictions))
