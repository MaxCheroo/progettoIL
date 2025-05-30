import gradio as gr
import joblib
import os
import subprocess

# Se mancano i file, esegui be.py per allenare il modello
if not (os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib")):
    print("Modello non trovato. Avvio script di training (be.py)...")
    subprocess.run(["python", "be.py"], check=True)
# Carica il modello e il vettorizzatore
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Funzione per predire il genere
def predici_genere(trama):
    print(trama)
    trama_trasformata = vectorizer.transform([trama])
    genere_predetto = model.predict(trama_trasformata)[0]
    return f"Genere predetto: {genere_predetto}"

# Interfaccia Gradio
interfaccia = gr.Interface(
    fn=predici_genere,
    inputs=gr.Textbox(lines=6, placeholder="Inserisci la trama del film qui..."),
    outputs=gr.Textbox(label="Genere Predetto"),
    title="Predizione Genere Film",
    description="Inserisci una trama e scopri a quale genere appartiene secondo il modello ML."
)

# Avvia l'app
interfaccia.launch()
>>>>>>> 44da763 (Upload progetto definitivo - ultima versione)