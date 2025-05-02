import gradio as gr
import requests

# Funzione per inviare il testo al backend Flask
def call_backend(text):
    response = requests.post("http://localhost:5000/tokenize", json={"text": text})
    if response.ok:
        return response.json()["tokens"]
    else:
        return ["Errore nel server."]

# UI Gradio
iface = gr.Interface(
    fn=call_backend,
    inputs=gr.Textbox(label="Inserisci una frase"),
    outputs=gr.Textbox(label="Token trovati"),
    title="Tokenizzatore di Frasi bellissime",
)

iface.launch()
