# Speech2Text with Whisper Transcriber

This project uses OpenAI's Whisper model to transcribe and translate audio files. The transcribed text is provided in multiple formats (TXT, JSON, SRT, TSV, VTT), and the text is also translated into French with timestamps.

## Installation

### Prerequisites
- Python 3.8+
- Install dependencies:

```bash
conda create --name <env> --file environment.yml
```
## Run the Streamlit app:
```bash
streamlit run app.py
```
## Features
- Audio file transcription with Whisper.
- Transcription available in multiple formats (TXT, JSON, SRT, TSV, VTT).
- Translation of the transcription to French with timestamps.




