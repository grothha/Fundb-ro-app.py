import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# --- PFAD-KONFIGURATION ---
# Wir nutzen den absoluten Pfad, damit das Modell auch auf Servern sicher gefunden wird
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_file():
    # Prüfen, ob die Datei existiert, bevor wir laden
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modell nicht gefunden unter: {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_labels():
    if not os.path.exists(LABEL_PATH):
        st.error(f"Labels nicht gefunden unter: {LABEL_PATH}")
        return []
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        # Entfernt Index-Zahlen (z.B. "0 Jacke"), falls vorhanden
        labels = [line.strip() for line in f.readlines()]
    return labels

# --- UI SETUP ---
st.set_page_config(page_title="Schul-Fundbüro KI", page_icon="🏫")
st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild hoch und die KI erkennt die Kategorie.")

# Modell und Labels initialisieren
model = load_model_file()
labels = load_labels()

# --- BILD-UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bildvorbereitung (Preprocessing)
    # 1. Größe anpassen (224x224 ist Standard für viele Keras-Modelle)
    img = image.resize((224, 224))
    img_array = np.asarray(img)
    
    # 2. Normalisierung (Wichtig: Muss exakt wie beim Training sein!)
    # Viele Teachable Machine Modelle nutzen (x / 127.5) - 1
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # 3. Vorhersage
    with st.spinner('KI analysiert...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]

    # 4. Ergebnis anzeigen
    st.divider()
    st.subheader("🔎 Ergebnis:")
    
    if labels:
        label_name = labels[index]
        st.metric(label="Kategorie", value=label_name)
        st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")
        
        # Fortschrittsbalken zur Visualisierung
        st.progress(float(confidence_score))
    else:
        st.warning("Labels konnten nicht geladen werden.")
    st.subheader("🔎 Ergebnis:")
    st.write(f"**Kategorie:** {labels[index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.2f}%")
