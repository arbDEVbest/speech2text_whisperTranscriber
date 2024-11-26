import streamlit as st
from src.whisper_transcriber import WhisperTranscriber, WhisperModelLoader
import os
import re
import json
from pathlib import Path

# CSS personnalisé pour styliser le bouton de téléchargement
st.markdown(
    """
    <style>
    .main {
    max-width: 100%; /* Largeur maximale de la page */
    margin: 0 auto; /* Centrer le contenu */
    }
    .stDownloadButton > button {
        width: 50%;
        background-color: #4CAF50; /* Couleur de fond */
        color: white; /* Couleur du texte */
        border: none; /* Suppression de la bordure */
        padding: 10px 20px; /* Espacement interne */
        text-align: center; /* Centrage du texte */
        text-decoration: none; /* Suppression du soulignement */
        display: inline-block; /* Style de bloc en ligne */
        font-size: 16px; /* Taille de la police */
        margin: 4px 2px; /* Marge */
        cursor: pointer; /* Curseur de pointeur */
        border-radius: 8px; /* Bords arrondis */
        transition-duration: 0.4s; /* Durée de la transition pour l’effet */
    }

    .stDownloadButton > button:hover {
        background-color: #45a049; /* Couleur de fond au survol */
        color: white; /* Couleur du texte au survol */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize the transcriber with model loader strategy
transcriber = WhisperTranscriber(model_loader=WhisperModelLoader(), model_name="base")

# Interface utilisateur
st.title("Transcription et traduction audio avec Whisper")
st.write("Chargez un fichier audio pour générer la transcription et sa traduction.")

# Chargement du fichier audio
uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
     # Récupérer le nom du fichier
    file_name = re.sub(r" - |- | -| ","-",uploaded_file.name)
    file_name_stem = Path(file_name).stem

    # Enregistrement temporaire du fichier audio
    with open(f"audio/{file_name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Spécification du répertoire de sortie
    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transcription et traduction
    with st.spinner("Transcription et traduction en cours..."):
        transcriptions = transcriber.transcribe_and_translate(f"audio/{file_name}", output_dir)

    # Afficher la transcription originale
    st.write("### Transcription originale :")
    st.text_area("Texte", transcriptions["txt"], height=200)   

   # Traduction en français
    st.write("### Traduction en français :")
    st.text_area("Texte traduit", transcriptions["translated_txt"], height=200)   
    
    # Télécharger les fichiers de sortie
    st.write("### Téléchargements :")

    st.download_button("Télécharger la transcription (TXT)", data=transcriptions["txt"], file_name=f"{file_name_stem}.txt")
    st.download_button("Télécharger la transcription (JSON)", data=json.dumps(transcriptions["json"], indent=2), file_name=f"{file_name_stem}.json")
    st.download_button("Télécharger la transcription (SRT)", data=transcriptions["srt"], file_name=f"{file_name_stem}.srt")
    st.download_button("Télécharger la transcription (TSV)", data=transcriptions["tsv"], file_name=f"{file_name_stem}.tsv")
    st.download_button("Télécharger la transcription (VTT)", data=transcriptions["vtt"], file_name=f"{file_name_stem}.vtt")
    st.download_button("Télécharger la transcription  en français (TXT)", data=transcriptions["translated_txt"], file_name=f"{file_name_stem}_translated.txt")