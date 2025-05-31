import gradio as gr
from model_backend import predici_genere, estrai_feature_rilevanti

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
