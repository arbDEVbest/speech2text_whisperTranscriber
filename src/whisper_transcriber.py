import whisper
from googletrans import Translator
import os
import json
from pathlib import Path
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    """
    Abstract class for model loading strategy
    """
    @abstractmethod
    def load_model(self, model_name: str, model_dir: str):
        pass

class WhisperModelLoader(ModelLoader):
    """
    Concrete implementation of ModelLoader for Whisper models.
    """
    def load_model(self, model_name: str, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        
        if not os.path.exists(model_path):
            print("Downloading the model...")
            return whisper.load_model(model_name, download_root=model_dir)
        else:
            print(f"Model already exists at {model_path}")
            return whisper.load_model(model_name, download_root=model_dir)    

class WhisperTranscriber:
    def __init__(self, model_loader: ModelLoader, model_name: str = "base", model_dir: str = "models"):
        """
        Initialize the WhisperTranscriber with a specific Whisper model and translation strategy.
        
        :param model_loader: Strategy for loading models.
        :param model_name: The model size to load (e.g., 'base', 'small', 'medium', 'large')
        :param model_dir: Directory for saving and loading models.
        """
        self.model_loader = model_loader
        self.model_name = model_name
        self.model_dir = model_dir
        self.translator = Translator()
        self.model = self.model_loader.load_model(self.model_name, self.model_dir)

    def transcribe_and_translate(self, audio_file_path: str, output_dir: str) -> None:
        """
        Transcribes the audio file and saves the outputs in various formats.
        Also translates the transcribed text into French and saves it in a specified format.
        
        :param audio_file_path: Path to the audio file to be transcribed.
        :param output_dir: Directory where the output files will be saved.
        """
        print(f"audio_file_path ===> {audio_file_path}")
        result = self.model.transcribe(audio_file_path)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the original transcription in different formats
        contents = self.save_transcription(result, output_dir, audio_file_path)
        
        # Generate output in different formats
        output = {
            "json": result,
            "txt": result['text'],
            "srt": contents["srt"],
            "tsv": contents["tsv"],
            "vtt": contents["vtt"]
        }
        # Translate text to French with timestamps
        translated_text_with_timestamps = self.translate_text_with_timestamps(result)

        output["translated_txt"] = translated_text_with_timestamps
        file_audio_name = Path(audio_file_path)
        # Save the translated text with timestamps
        path_audio_dir = Path(f"{output_dir}/{file_audio_name.stem}")
        path_audio_dir.mkdir(exist_ok=True, parents=True)
        translated_output_path = path_audio_dir / f"{file_audio_name.stem}_translated.txt"
        with open(translated_output_path, "w", encoding="utf-8") as f:
            f.write(translated_text_with_timestamps)
        return output    

    def save_transcription(self, result, output_dir: str, path_file_audio_str: str) -> None:
        """
        Save transcription results in JSON, SRT, TSV, TXT, and VTT formats.
        
        :param result: The transcription result from Whisper.
        :param output_dir: Directory where the output files will be saved.
        """
        # Convert path file audio from str to Path
        path_file_audio = Path(path_file_audio_str)
        path_audio_dir = Path(f"{output_dir}/{path_file_audio.stem}")
        path_audio_dir.mkdir(exist_ok=True, parents=True)
        # Save in JSON
        with open(path_audio_dir / f"{path_file_audio.stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        # Save in TXT
        with open(path_audio_dir / f"{path_file_audio.stem}.txt", "w", encoding="utf-8") as f:
            f.write(result['text'])
        
        # Save in SRT and VTT formats with timestamps
        srt_content = ""
        vtt_content = "WEBVTT\n\n"
        for i, segment in enumerate(result['segments']):
            start = self.format_timestamp(segment['start'])
            end = self.format_timestamp(segment['end'])
            text = segment['text'].strip()
            srt_content += f"{i + 1}\n{start} --> {end}\n{text}\n\n"
            vtt_content += f"{start} --> {end}\n{text}\n\n"

        with open(path_audio_dir / f"{path_file_audio.stem}.srt", "w", encoding="utf-8") as f:
            f.write(srt_content)
        with open(path_audio_dir / f"{path_file_audio.stem}.vtt", "w", encoding="utf-8") as f:
            f.write(vtt_content)
        
        # Save in TSV
        tsv_content = "start\tend\ttext\n"
        for segment in result['segments']:
            start = self.format_timestamp(segment['start'], sep=".")
            end = self.format_timestamp(segment['end'], sep=".")
            text = segment['text'].strip().replace("\t", " ")
            tsv_content += f"{start}\t{end}\t{text}\n"
        
        with open(path_audio_dir / f"{path_file_audio.stem}.tsv", "w", encoding="utf-8") as f:
            f.write(tsv_content)

        contents = {
            "srt": srt_content,
            "tsv": tsv_content,
            "vtt": vtt_content
        }    
        return contents

    def translate_text_with_timestamps(self, result) -> str:
        """
        Translates each segment of the transcription to French, keeping timestamps.
        
        :param result: The transcription result from Whisper.
        :return: Translated text with timestamps as a formatted string.
        """
        translated_text_with_timestamps = ""
        
        for segment in result['segments']:
            start = self.format_timestamp(segment['start'])
            end = self.format_timestamp(segment['end'])
            text = segment['text'].strip()
            translated_text = self.translator.translate(text, dest="fr").text
            translated_text_with_timestamps += f"{start} --> {end}\n{translated_text}\n\n"
        
        return translated_text_with_timestamps

    @staticmethod
    def format_timestamp(seconds: float, sep: str = ",") -> str:
        """
        Formats seconds into a timestamp suitable for SRT/VTT formats.
        
        :param seconds: The timestamp in seconds.
        :param sep: Separator for the milliseconds ("," for SRT, "." for VTT).
        :return: Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}{sep}{millis:03}"
