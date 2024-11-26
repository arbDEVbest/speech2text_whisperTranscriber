import pandas as pd
from transformers import pipeline
from gtts import gTTS
import os
import re

# Charger le fichier TSV
file_path = "output_files/Conclusion/Conclusion.tsv"
data = pd.read_csv(file_path, sep='\t')  # Lire le fichier TSV avec pandas

# Initialiser le modèle de traduction si les phrases ne sont pas déjà traduites
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# Créer un dossier pour sauvegarder les fichiers audio
output_folder = "audio_outputs"
os.makedirs(output_folder, exist_ok=True)

# Fonction pour traduire et générer l'audio
def translate_and_tts(row):
    english_text = row['text']
    
    french_text = translator(english_text)[0]['translation_text']
    
    # Générer l'audio en français
    tts = gTTS(french_text, lang="fr")
    seg += 1
    output_path = os.path.join(output_folder, f"segment-{seg}.mp3")
    tts.save(output_path)
    
    print(f"Audio généré pour: {english_text} -> {french_text}")
    return french_text

# Appliquer la fonction de traduction et TTS
data['french_text'] = data.apply(translate_and_tts, axis=1)

# Sauvegarder les traductions mises à jour dans un nouveau fichier TSV
data.to_csv("fichier_traduit.tsv", sep='\t', index=False)
print("Processus terminé : Les fichiers audio sont sauvegardés et le fichier TSV mis à jour.")
