from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import os
import io
import base64 # For encoding audio data

# For ElevenLabs
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
# from elevenlabs import play # We won't use play directly, we'll stream/send bytes

# Load environment variables (for ELEVENLABS_API_KEY)
load_dotenv()

# --- Constants and Setup (from notebook) ---
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
FRAME_LEN = 128

# --- Load auxiliary files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_MAP_PATH = os.path.join(BASE_DIR, "character_to_prediction_index.json")
INFERENCE_ARGS_PATH = os.path.join(BASE_DIR, "inference_args.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

# Load character map
try:
    with open(CHAR_MAP_PATH, "r") as f:
        char_to_num_orig = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Character map file not found at {CHAR_MAP_PATH}")

char_to_num = char_to_num_orig.copy()

pad_token = 'P'
start_token = '<'
end_token = '>'
current_max_idx = -1
if char_to_num:
    current_max_idx = max(char_to_num.values())

pad_token_idx = char_to_num.get(pad_token, current_max_idx + 1)
if pad_token not in char_to_num: char_to_num[pad_token] = pad_token_idx; current_max_idx +=1
start_token_idx = char_to_num.get(start_token, current_max_idx + 1)
if start_token not in char_to_num: char_to_num[start_token] = start_token_idx; current_max_idx +=1
end_token_idx = char_to_num.get(end_token, current_max_idx + 1)
if end_token not in char_to_num: char_to_num[end_token] = end_token_idx

num_to_char = {j: i for i, j in char_to_num.items()}

try:
    with open(INFERENCE_ARGS_PATH, "r") as f:
        infargs = json.load(f)
    FEATURE_COLUMNS = infargs["selected_columns"]
except FileNotFoundError:
    raise RuntimeError(f"Inference args file not found at {INFERENCE_ARGS_PATH}")

# --- ElevenLabs Client Initialization ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        # You can test the API key here if needed, e.g., by fetching voices
        # elevenlabs_client.voices.get_all() 
    except Exception as e:
        print(f"Warning: Could not initialize ElevenLabs client (check API key or service): {e}")
        elevenlabs_client = None
else:
    print("Warning: ELEVENLABS_API_KEY not found in .env file. Text-to-speech will be disabled.")


# --- TFLite Model Loading ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    prediction_fn = interpreter.get_signature_runner("serving_default")
except Exception as e:
    raise RuntimeError(f"Error loading TFLite model: {e}")

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

class LandmarkFrame(BaseModel):
    landmarks: List[float]

class LiveDataInput(BaseModel):
    frames: List[LandmarkFrame]

async def generate_speech_audio(text_to_speak: str):
    """Generates speech audio using ElevenLabs and returns base64 encoded MP3."""
    if not elevenlabs_client or not text_to_speak.strip():
        return None
    try:
        # The convert method returns an iterator of bytes
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=text_to_speak,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with your desired voice_id
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128", # Common MP3 format
        )
        
        # Accumulate bytes from the iterator
        audio_bytes_list = []
        for chunk in audio_stream:
            if chunk: # Ensure chunk is not None
                 audio_bytes_list.append(chunk)
        
        if not audio_bytes_list:
            print("ElevenLabs returned no audio data.")
            return None

        audio_bytes = b"".join(audio_bytes_list)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"Error generating speech with ElevenLabs: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(os.path.join(BASE_DIR, "static", "index.html"), "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/predict_parquet")
async def predict_parquet(file: UploadFile = File(...)):
    if not file.filename.endswith('.parquet'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a Parquet file.")
    try:
        contents = await file.read()
        parquet_file = io.BytesIO(contents)
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Parquet file: {e}")

    if not all(col in df.columns for col in FEATURE_COLUMNS):
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        raise HTTPException(status_code=400, detail=f"Uploaded Parquet file is missing columns: {', '.join(missing_cols)}")
    
    try:
        landmark_data = df[FEATURE_COLUMNS].values.astype(np.float32)
        if landmark_data.ndim != 2 or landmark_data.shape[1] != len(FEATURE_COLUMNS):
             raise ValueError(f"Input data should have {len(FEATURE_COLUMNS)} columns. Got shape {landmark_data.shape}.")
        if landmark_data.shape[0] == 0:
             raise ValueError("Input data has 0 frames.")

        output = prediction_fn(inputs=landmark_data)
        prediction_logits = output['outputs']
        predicted_indices = np.argmax(prediction_logits, axis=1)
        prediction_str = "".join([num_to_char.get(int(idx_val), "") for idx_val in predicted_indices])
        
        audio_base64 = await generate_speech_audio(prediction_str)
            
        return {"prediction": prediction_str, "audio_base64": audio_base64}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Prediction error (Parquet): {e}") # Log detailed error
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_live_data")
async def predict_live_data(live_data: LiveDataInput):
    try:
        frames_data = [frame.landmarks for frame in live_data.frames]
        if not frames_data:
            raise ValueError("Received empty landmark frames.")

        landmark_data_np = np.array(frames_data, dtype=np.float32)

        if landmark_data_np.ndim != 2 or landmark_data_np.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Live landmark data has incorrect shape. Expected [N, {len(FEATURE_COLUMNS)}], got {landmark_data_np.shape}")
        if landmark_data_np.shape[0] == 0:
             raise ValueError("Input data has 0 frames.")

        output = prediction_fn(inputs=landmark_data_np)
        prediction_logits = output['outputs']
        
        predicted_indices = np.argmax(prediction_logits, axis=1)
        prediction_str = "".join([num_to_char.get(int(idx_val), "") for idx_val in predicted_indices])

        audio_base64 = await generate_speech_audio(prediction_str)

        return {"prediction": prediction_str, "audio_base64": audio_base64}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Prediction error (Live Data): {e}") # Log detailed error
        raise HTTPException(status_code=500, detail=f"Error during live data prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("FastAPI app ready. Run with: uvicorn main:app --reload")
    print(f"Access the UI at http://127.0.0.1:8000")
    if not ELEVENLABS_API_KEY:
        print("\nWARNING: ELEVENLABS_API_KEY is not set. Text-to-speech functionality will be disabled.")
        print("Please create a .env file with your API key (e.g., ELEVENLABS_API_KEY=yourkey) to enable TTS.\n")