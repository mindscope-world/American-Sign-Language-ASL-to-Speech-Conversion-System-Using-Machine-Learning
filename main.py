from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # If you want to serve HTML from templates
from pydantic import BaseModel # For request body validation
from typing import List, Dict, Any
import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import os
import io

# --- Constants and Setup (from notebook) ---
# These definitions are for understanding the model's internal preprocessing.
# The TFLite model from TFLiteModel class already encapsulates this.
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
FRAME_LEN = 128 # Relevant for model's internal pre_process logic

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

char_to_num = char_to_num_orig.copy() # Work with a copy

pad_token = 'P'
start_token = '<'
end_token = '>'

# Ensure special tokens are added correctly and consistently
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

# Load inference_args.json for FEATURE_COLUMNS
try:
    with open(INFERENCE_ARGS_PATH, "r") as f:
        infargs = json.load(f)
    FEATURE_COLUMNS = infargs["selected_columns"] # Should be 156 features
except FileNotFoundError:
    raise RuntimeError(f"Inference args file not found at {INFERENCE_ARGS_PATH}")

# --- TFLite Model Loading ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    prediction_fn = interpreter.get_signature_runner("serving_default")
except Exception as e:
    raise RuntimeError(f"Error loading TFLite model: {e}")

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Pydantic model for live data input
class LandmarkFrame(BaseModel):
    landmarks: List[float] # A flat list of 156 landmark coordinates

class LiveDataInput(BaseModel):
    frames: List[LandmarkFrame]

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
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
        
        return {"prediction": prediction_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_live_data")
async def predict_live_data(live_data: LiveDataInput):
    """
    Receives a sequence of landmark frames (already pre-formatted by client)
    and returns the ASL fingerspelling prediction.
    """
    try:
        frames_data = [frame.landmarks for frame in live_data.frames]
        if not frames_data:
            raise ValueError("Received empty landmark frames.")

        # Convert to NumPy array, ensuring correct shape [num_frames, num_features]
        landmark_data_np = np.array(frames_data, dtype=np.float32)

        if landmark_data_np.ndim != 2 or landmark_data_np.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Live landmark data has incorrect shape. Expected [N, {len(FEATURE_COLUMNS)}], got {landmark_data_np.shape}")
        if landmark_data_np.shape[0] == 0:
             raise ValueError("Input data has 0 frames.")

        # Run inference
        # The TFLite model's __call__ from TFLiteModel handles preprocessing.
        output = prediction_fn(inputs=landmark_data_np)
        prediction_logits = output['outputs']
        
        # Decode prediction
        predicted_indices = np.argmax(prediction_logits, axis=1)
        prediction_str = "".join([num_to_char.get(int(idx_val), "") for idx_val in predicted_indices])
        
        return {"prediction": prediction_str}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during live data prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("FastAPI app ready. Run with: uvicorn main:app --reload")
    print(f"Access the UI at http://127.0.0.1:8000")