from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # <-- sostituito
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib


# Import dataset
df = pd.read_csv("movie_plots_filtered.csv")
print("dataset importato")

# Filtraggio dei dati
df_clean = df[['plot', 'genre']].dropna()
df_clean = df_clean[df_clean['genre'].str.lower() != 'unknown']
print("dataset filtrato")

# Rinomino le colonne per chiarezza
df_clean.columns = ['plot', 'genre']
print("rinonimo due colonne")

# Creazione delle due versioni dei TF-IDF vectorizer
vectorizer_with_stopwords = TfidfVectorizer()
vectorizer_without_stopwords = TfidfVectorizer(stop_words='english')

print("creazione versione TF-IDF")

# Creazione delle due matrici
X_with = vectorizer_with_stopwords.fit_transform(df_clean['plot'])
X_without = vectorizer_without_stopwords.fit_transform(df_clean['plot'])
joblib.dump(vectorizer_without_stopwords, "vectorizer.joblib")
y = df_clean['genre']
print("creazione matrici")

# Suddivisione in training e test set
X_train_with, X_test_with, y_train, y_test = train_test_split(X_with, y, test_size=0.2, random_state=42)
X_train_without, X_test_without, _, _ = train_test_split(X_without, y, test_size=0.2, random_state=42)
print("suddivisione in training e test set")

# Modello con stopwords
svc_with = LinearSVC()
svc_with.fit(X_train_with, y_train)
y_pred_with = svc_with.predict(X_test_with)
print("creazione modello con stopwords")

# Modello senza stopwords
svc_without = LinearSVC()
svc_without.fit(X_train_without, y_train)
y_pred_without = svc_without.predict(X_test_without)
joblib.dump(svc_without, "model.joblib")
print("creazione modello senza stopwords")

# Accuratezze
acc_with = accuracy_score(y_test, y_pred_with)
acc_without = accuracy_score(y_test, y_pred_without)
print("calcolo accuracy")

# Distribuzione dei generi
genre_counts = df_clean['genre'].value_counts().head(20)  # mostriamo i primi 20

# Report dettagliato
report_with = classification_report(y_test, y_pred_with, output_dict=True, zero_division=0)
report_without = classification_report(y_test, y_pred_without, output_dict=True, zero_division=0)

print("Report modello con stopwords")
report_df = pd.DataFrame(report_with).transpose()
print(report_df)

print("Report modello senza stopwords")
report_df = pd.DataFrame(report_without).transpose()
print(report_df)

# Matrici di confusione
conf_matrix_with = confusion_matrix(y_test, y_pred_with)
conf_matrix_without = confusion_matrix(y_test, y_pred_without)

# Trovare quali classi non vengono mai predette - modello senza stopwords
classes_predette = np.unique(y_pred_without)
classi_test = np.unique(y_test)
classi_non_predette = set(classi_test) - set(classes_predette)
print("Classi presenti nel test ma mai predette:", classi_non_predette)

# Trovare quali classi non vengono mai predette - modello con stopwords
classes_predette = np.unique(y_pred_with)
classi_test = np.unique(y_test)
classi_non_predette = set(classi_test) - set(classes_predette)
print("Classi presenti nel test ma mai predette:", classi_non_predette)

# Mostra risultati e grafici
results_df = pd.DataFrame({
    "Configurazione": ["Con Stopwords", "Senza Stopwords"],
    "Accuratezza": [acc_with, acc_without]
})

print(results_df)

# Grafico: distribuzione dei generi
#plt.figure(figsize=(10, 6))
#sns.barplot(x=genre_counts.values, y=genre_counts.index)
#plt.title("Distribuzione dei generi più frequenti")
#plt.xlabel("Numero di film")
#plt.ylabel("Genere")
#plt.tight_layout()
#plt.show()


