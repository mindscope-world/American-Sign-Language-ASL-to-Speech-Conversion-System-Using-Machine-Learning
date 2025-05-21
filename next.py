from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
FRAME_LEN = 128 # This was used for training and is relevant for model's internal pre_process

# --- Load auxiliary files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_MAP_PATH = os.path.join(BASE_DIR, "character_to_prediction_index.json")
INFERENCE_ARGS_PATH = os.path.join(BASE_DIR, "inference_args.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

# Load character map
try:
    with open(CHAR_MAP_PATH, "r") as f:
        char_to_num = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Character map file not found at {CHAR_MAP_PATH}")

# Add special tokens if not already present (idempotent)
pad_token = 'P'
start_token = '<'
end_token = '>'
# Determine next available indices
current_max_idx = -1
if char_to_num:
    current_max_idx = max(char_to_num.values())

pad_token_idx = char_to_num.get(pad_token, current_max_idx + 1)
if pad_token not in char_to_num: current_max_idx +=1
start_token_idx = char_to_num.get(start_token, current_max_idx + 1)
if start_token not in char_to_num: current_max_idx +=1
end_token_idx = char_to_num.get(end_token, current_max_idx + 1)

char_to_num[pad_token] = pad_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j: i for i, j in char_to_num.items()}

# Load inference_args.json for FEATURE_COLUMNS
try:
    with open(INFERENCE_ARGS_PATH, "r") as f:
        infargs = json.load(f)
    FEATURE_COLUMNS = infargs["selected_columns"]
except FileNotFoundError:
    raise RuntimeError(f"Inference args file not found at {INFERENCE_ARGS_PATH}")


# --- TFLite Model Loading ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    # interpreter.allocate_tensors() # Not strictly necessary here
    prediction_fn = interpreter.get_signature_runner("serving_default")
except Exception as e:
    raise RuntimeError(f"Error loading TFLite model: {e}")


# --- FastAPI App Initialization ---
app = FastAPI()

# Mount static files directory (for HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
    try:
        with open(os.path.join(BASE_DIR, "static", "index.html"), "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives a Parquet file, processes it, and returns the ASL fingerspelling prediction.
    """
    if not file.filename.endswith('.parquet'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a Parquet file.")

    try:
        contents = await file.read()
        parquet_file = io.BytesIO(contents) # Use io.BytesIO for pandas
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Parquet file: {e}")

    if not all(col in df.columns for col in FEATURE_COLUMNS):
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        raise HTTPException(status_code=400, detail=f"Uploaded Parquet file is missing columns: {', '.join(missing_cols)}")
    
    try:
        # Prepare landmark data for the model
        # The TFLiteModel's __call__ input_signature expects [None, len(FEATURE_COLUMNS)]
        landmark_data = df[FEATURE_COLUMNS].values.astype(np.float32)

        if landmark_data.ndim != 2 or landmark_data.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Input data should have {len(FEATURE_COLUMNS)} columns. Got shape {landmark_data.shape}.")
        if landmark_data.shape[0] == 0:
             raise ValueError("Input data has 0 frames. Please provide a sequence with at least one frame.")

        # Run inference
        output = prediction_fn(inputs=landmark_data)
        prediction_logits = output['outputs'] # This is one-hot encoded (depth 59)
        
        # Decode prediction
        predicted_indices = np.argmax(prediction_logits, axis=1)
        
        prediction_chars = []
        for idx_val in predicted_indices:
            char = num_to_char.get(int(idx_val), "") # Ensure idx_val is int for dict lookup
            prediction_chars.append(char)
        
        prediction_str = "".join(prediction_chars)
        
        return {"prediction": prediction_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run from the asl_fastapi_app directory
    # uvicorn main:app --reload
    print("FastAPI app ready. Run with: uvicorn main:app --reload")
    print(f"Access the UI at http://127.0.0.1:8000")
    