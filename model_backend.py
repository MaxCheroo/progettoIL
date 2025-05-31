# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Caricamento e preparazione dati
# df = pd.read_csv("datasetpronto.csv")
# testi = df['Cleaned_Plot'].tolist()
# etichette = df['Genre'].tolist()

# # Vettorizzazione senza stopwords
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(testi)

# # Addestramento modello migliore
# model = SVC(kernel='linear', C=1, probability=True)
# model.fit(X, etichette)
# classi = model.classes_
# coef = model.coef_.toarray()

# # Funzione di predizione
# def predici_genere(plot):
#     X_input = vectorizer.transform([plot])
#     pred = model.predict(X_input)[0]
#     return pred

# # Funzione per estrarre parole più influenti
# def estrai_feature_rilevanti(plot, top_n=10):
#     X_input = vectorizer.transform([plot])
#     pesi = X_input.dot(model.coef_.T).toarray().flatten()
#     classe_idx = np.argmax(pesi)
#     coeff = coef[classe_idx]
    
#     indici = X_input.nonzero()[1]
#     parole_input = vectorizer.get_feature_names_out()[indici]
#     pesi_input = coeff[indici]
    
#     top = np.argsort(np.abs(pesi_input))[-top_n:][::-1]
#     return list(zip(parole_input[top], pesi_input[top]))



import joblib
import numpy as np
import re
import string

# Funzione per pulire SOLO l'input utente
def pulisci_testo(testo):
    testo = testo.lower()
    testo = re.sub(r'\d+', '', testo)
    testo = testo.translate(str.maketrans('', '', string.punctuation))
    testo = testo.strip()
    return testo

model = joblib.load("modello_svc.joblib")
vectorizer = joblib.load("vectorizer_tfidf.joblib")
coef = model.coef_.toarray()
classi = model.classes_

def predici_genere(plot):
    plot = pulisci_testo(plot)
    X_input = vectorizer.transform([plot])
    pred = model.predict(X_input)[0]
    return pred

def estrai_feature_rilevanti(plot, top_n=10):
    plot = pulisci_testo(plot)
    X_input = vectorizer.transform([plot])
    pesi = X_input.dot(model.coef_.T).toarray().flatten()
    classe_idx = np.argmax(pesi)
    coeff = coef[classe_idx]
    
    indici = X_input.nonzero()[1]
    parole_input = vectorizer.get_feature_names_out()[indici]
    pesi_input = coeff[indici]
    
    top = np.argsort(np.abs(pesi_input))[-top_n:][::-1]
    return list(zip(parole_input[top], pesi_input[top]))

