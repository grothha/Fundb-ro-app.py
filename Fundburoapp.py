import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model():
    # Anstatt tf.keras.models.load_model('model.h5') zu nutzen,
    # definieren wir hier die Architektur manuell.
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)), # Beispiel-Input-Größe: 10
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Für binäre Klassifikation
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 1. Modell initialisieren
model = create_model()
print("Modell wurde erfolgreich im Speicher erstellt (ohne .h5 Datei).")

# 2. Beispiel-Daten (Platzhalter für deine echten Daten)
# Falls du Vorhersagen machen willst, braucht das Modell Gewichte.
# Ohne .h5 Datei sind die Gewichte zufällig, es sei denn, du trainierst kurz:
dummy_data = np.random.random((1, 10))

# 3. Vorhersage oder Training
prediction = model.predict(dummy_data)
print(f"Test-Vorhersage: {prediction}")

# --- PFAD-KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Falls die Dateien direkt im selben Ordner liegen, entferne das "model" im Pfad
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "labels.txt")

# --- FUNKTIONEN ---
@st.cache_resource
def load_model_file():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modell nicht gefunden unter: {MODEL_PATH}")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

def load_labels():
    if not os.path.exists(LABEL_PATH):
        st.error(f"Labels nicht gefunden unter: {LABEL_PATH}")
        return []
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        # Teachable Machine Labels sehen oft so aus: "0 Jacke"
        # Wir trennen die Zahl ab und nehmen nur den Namen.
        labels = []
        for line in f.readlines():
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels

# --- UI SETUP ---
st.set_page_config(page_title="Schul-Fundbüro KI", page_icon="🏫")
st.title("🏫 Schul-Fundbüro KI-App")
st.write("Lade ein Bild eines verlorenen Gegenstands hoch.")

# Modell und Labels laden
model = load_model_file()
labels = load_labels()

# --- BILD-UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("KI-Modell konnte nicht geladen werden. Bitte Pfade prüfen.")
    else:
        # Bild anzeigen
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

        # 1. Bildvorbereitung (Preprocessing)
        size = (224, 224)
        img = image.resize(size)
        img_array = np.asarray(img)
        
        # 2. Normalisierung (Standard für Teachable Machine)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        # 3. Vorhersage
        with st.spinner('KI analysiert das Fundstück...'):
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

        # 4. Ergebnis anzeigen
        st.divider()
        st.subheader("🔎 Analyse-Ergebnis")
        
        if labels and index < len(labels):
            label_name = labels[index]
            
            # Schöne Darstellung mit Spalten
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Gegenstand", value=label_name)
            with col2:
                st.metric(label="Sicherheit", value=f"{confidence_score * 100:.1f}%")
            
            # Fortschrittsbalken
            st.progress(float(confidence_score))
            
            if confidence_score < 0.6:
                st.warning("Hinweis: Die KI ist sich unsicher. Bitte manuell prüfen.")
        else:
            st.warning("Das Modell hat ein Ergebnis geliefert, aber die Labels fehlen.")

# --- FOOTER ---
st.sidebar.info("Tipp: Achte auf gute Beleuchtung beim Foto.")
