import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# DEBUG AUSGABE 👇 HIER EINFÜGEN
st.write("Aktuelles Verzeichnis:", os.getcwd())
st.write("Dateien im Hauptordner:", os.listdir())

# Titel der App
st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild hoch und die KI erkennt die Kategorie.")
# Titel der App
st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild hoch und die KI erkennt die Kategorie.")

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
    return model

model = load_model()

# Labels laden
def load_labels():
    with open("model/labels.txt", "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    return labels

labels = load_labels()

# Bild Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    img = image.resize((224, 224))
    img_array = np.asarray(img)
    img_array = img_array.astype(np.float32) / 127.5 - 1
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    st.subheader("🔎 Ergebnis:")
    st.write(f"**Kategorie:** {labels[index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.2f}%")
