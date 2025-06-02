import gradio as gr
from backend import predici_genere, estrai_feature_rilevanti


# # Se mancano i file, esegui be.py per allenare il modello
# if not (os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib")):
#     print("Modello non trovato. Avvio script di training (be.py)...")
#     subprocess.run(["python", "be.py"], check=True)

def interfaccia(plot):
    genere = predici_genere(plot)
    features = estrai_feature_rilevanti(plot)
    spiegazione = "\n".join([f"{f[0]} (peso: {f[1]:.4f})" for f in features])
    return f"Genere previsto: {genere}", spiegazione

gr.Interface(
    fn=interfaccia,
    inputs=gr.Textbox(lines=8, placeholder="Inserisci il plot del film..."),
    outputs=["text", "text"],
    title="Classificazione Genere Film",
    description="Inserisci una trama di film: il sistema predice il genere e mostra le parole più influenti"
).launch()
