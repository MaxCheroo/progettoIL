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

