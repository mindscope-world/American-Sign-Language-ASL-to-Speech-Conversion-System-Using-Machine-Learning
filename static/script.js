document.addEventListener('DOMContentLoaded', () => {
    // Mode switching
    const uploadModeBtn = document.getElementById('uploadModeBtn');
    const webcamModeBtn = document.getElementById('webcamModeBtn');
    const uploadSection = document.getElementById('uploadSection');
    const webcamSection = document.getElementById('webcamSection');

    uploadModeBtn.addEventListener('click', () => {
        uploadSection.classList.add('active-content');
        webcamSection.classList.remove('active-content');
        uploadModeBtn.classList.add('active');
        webcamModeBtn.classList.remove('active');
        stopWebcam(); // Stop webcam if switching modes
    });

    webcamModeBtn.addEventListener('click', () => {
        webcamSection.classList.add('active-content');
        uploadSection.classList.remove('active-content');
        webcamModeBtn.classList.add('active');
        uploadModeBtn.classList.remove('active');
    });

    // Parquet Upload Elements
    const parquetFileInput = document.getElementById('parquetFile');
    const predictParquetButton = document.getElementById('predictParquetButton');
    
    // Webcam Elements
    const videoElement = document.getElementById('webcamVideo');
    const canvasElement = document.getElementById('webcamCanvas');
    const canvasCtx = canvasElement.getContext('2d');
    const startWebcamButton = document.getElementById('startWebcamButton');
    const stopWebcamButton = document.getElementById('stopWebcamButton');
    const capturePredictButton = document.getElementById('capturePredictButton');

    // Shared Elements
    const predictionResult = document.getElementById('predictionResult');
    const errorMessage = document.getElementById('errorMessage');
    const statusMessage = document.getElementById('statusMessage');
    const resultsSection = document.getElementById('resultsSection');

    let hands, pose, camera;
    let landmarkFrames = [];
    let isCapturing = false;
    const CAPTURE_DURATION_MS = 3000; // Capture for 3 seconds
    const CAPTURE_FPS = 20; // Aim for ~20 FPS for landmark collection
    const CAPTURE_FRAMES_COUNT = CAPTURE_DURATION_MS / 1000 * CAPTURE_FPS;
    let captureIntervalId = null;

    // --- Feature Column Mapping (from notebook) ---
    // This needs to match the order in FEATURE_COLUMNS from inference_args.json
    // LPOSE and RPOSE are as defined in the notebook
    const LPOSE_INDICES = [13, 15, 17, 19, 21];
    const RPOSE_INDICES = [14, 16, 18, 20, 22];
    const POSE_LANDMARK_INDICES = [...LPOSE_INDICES, ...RPOSE_INDICES]; // 10 pose landmarks
    const NUM_HAND_LANDMARKS = 21;
    const NUM_POSE_LANDMARKS_TO_EXTRACT = POSE_LANDMARK_INDICES.length; // Should be 10
    const TOTAL_FEATURES_PER_COORD = NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS_TO_EXTRACT; // 21+21+10 = 52
    const TOTAL_FEATURES = TOTAL_FEATURES_PER_COORD * 3; // x, y, z for each, so 52 * 3 = 156

    // --- MediaPipe Setup ---
    function onResults(results) {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        const frameLandmarks = new Array(TOTAL_FEATURES).fill(NaN);

        // Process Pose landmarks
        if (results.poseLandmarks) {
            drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
            drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
            
            POSE_LANDMARK_INDICES.forEach((poseIdx, i) => {
                if (results.poseLandmarks[poseIdx]) {
                    frameLandmarks[NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].x; // x_pose_i
                    frameLandmarks[TOTAL_FEATURES_PER_COORD + NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].y; // y_pose_i
                    frameLandmarks[TOTAL_FEATURES_PER_COORD * 2 + NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].z; // z_pose_i
                }
            });
        }

        // Process Hand landmarks
        if (results.multiHandLandmarks) {
            for (let handIndex = 0; handIndex < results.multiHandLandmarks.length; handIndex++) {
                const landmarks = results.multiHandLandmarks[handIndex];
                const classification = results.multiHandedness[handIndex];
                const isRightHand = classification.label === 'Right'; // MediaPipe gives 'Left'/'Right' based on image content, not user's perspective

                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: isRightHand ? '#00FF00' : '#FF0000', lineWidth: 3 });
                drawLandmarks(canvasCtx, landmarks, { color: isRightHand ? '#FFFFFF' : '#0000FF', lineWidth: 1, radius: 3 });

                const offset = isRightHand ? 0 : NUM_HAND_LANDMARKS;
                landmarks.forEach((landmark, i) => {
                    frameLandmarks[offset + i] = landmark.x; // x_right_hand_i or x_left_hand_i
                    frameLandmarks[TOTAL_FEATURES_PER_COORD + offset + i] = landmark.y; // y_...
                    frameLandmarks[TOTAL_FEATURES_PER_COORD * 2 + offset + i] = landmark.z; // z_...
                });
            }
        }
        canvasCtx.restore();

        if (isCapturing && landmarkFrames.length < CAPTURE_FRAMES_COUNT) {
            landmarkFrames.push({ landmarks: frameLandmarks });
        }
    }

    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    hands.onResults(onResults); // Re-use onResults for now, can be split later

    pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    pose.onResults(onResults);
    
    async function processFrame() {
        if (!videoElement.paused && !videoElement.ended) {
            await hands.send({ image: videoElement });
            await pose.send({ image: videoElement }); // Send same frame to pose
        }
        requestAnimationFrame(processFrame);
    }
    
    // --- Event Listeners ---
    if (predictParquetButton) {
        predictParquetButton.addEventListener('click', async () => {
            const file = parquetFileInput.files[0];
            if (!file) {
                errorMessage.textContent = 'Please select a Parquet file.';
                predictionResult.textContent = '---';
                resultsSection.style.display = 'none';
                return;
            }
            performPrediction(file, '/predict_parquet');
        });
    }

    if (startWebcamButton) {
        startWebcamButton.addEventListener('click', async () => {
            try {
                statusMessage.textContent = 'Starting webcam...';
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                videoElement.srcObject = stream;
                videoElement.onloadedmetadata = () => {
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    camera = new Camera(videoElement, {
                        onFrame: async () => {
                            // This will be handled by processFrame to avoid duplicate sends
                        },
                        width: 640,
                        height: 480
                    });
                    camera.start(); // Start the camera utility
                    processFrame(); // Start our combined processing loop
                    statusMessage.textContent = 'Webcam started. Position your hands.';
                };
                startWebcamButton.disabled = true;
                stopWebcamButton.disabled = false;
                capturePredictButton.disabled = false;
                errorMessage.textContent = '';
            } catch (err) {
                console.error("Error accessing webcam:", err);
                errorMessage.textContent = 'Error accessing webcam. Please ensure permissions are granted.';
                statusMessage.textContent = '';
            }
        });
    }
    
    function stopWebcam() {
        if (camera) {
            camera.stop(); // Stops MediaPipe's Camera utility
        }
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
        if(captureIntervalId) clearInterval(captureIntervalId);
        captureIntervalId = null;
        isCapturing = false;
        startWebcamButton.disabled = false;
        stopWebcamButton.disabled = true;
        capturePredictButton.disabled = true;
        statusMessage.textContent = 'Webcam stopped.';
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height); // Clear canvas
    }

    if (stopWebcamButton) {
        stopWebcamButton.addEventListener('click', stopWebcam);
    }

    if (capturePredictButton) {
        capturePredictButton.addEventListener('click', () => {
            if (!videoElement.srcObject) {
                errorMessage.textContent = "Webcam is not active. Please start it first.";
                return;
            }
            isCapturing = true;
            landmarkFrames = [];
            statusMessage.textContent = `Capturing landmarks for ${CAPTURE_DURATION_MS / 1000} seconds...`;
            capturePredictButton.disabled = true; // Disable during capture

            // Use setTimeout to stop capturing after DURATION and then predict
            setTimeout(async () => {
                isCapturing = false;
                statusMessage.textContent = 'Capture complete. Predicting...';
                if (landmarkFrames.length > 0) {
                    const liveDataPayload = { frames: landmarkFrames };
                    await performPrediction(liveDataPayload, '/predict_live_data');
                } else {
                    errorMessage.textContent = "No landmarks captured. Try again.";
                    statusMessage.textContent = 'Webcam ready.';
                }
                capturePredictButton.disabled = false; // Re-enable after prediction
            }, CAPTURE_DURATION_MS);
        });
    }

    async function performPrediction(payload, endpoint) {
        predictionResult.textContent = 'Processing...';
        resultsSection.style.display = 'block';
        errorMessage.textContent = '';
        
        // Disable buttons during processing
        if(predictParquetButton) predictParquetButton.disabled = true;
        if(capturePredictButton) capturePredictButton.disabled = true;


        let requestOptions;
        if (payload instanceof File) { // For Parquet file
            const formData = new FormData();
            formData.append('file', payload);
            requestOptions = { method: 'POST', body: formData };
        } else { // For live data (JSON)
            requestOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };
        }

        try {
            const response = await fetch(endpoint, requestOptions);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }
            predictionResult.textContent = data.prediction || 'No prediction received.';
            statusMessage.textContent = 'Prediction complete.';

        } catch (error) {
            console.error('Error:', error);
            predictionResult.textContent = '---';
            errorMessage.textContent = `Error: ${error.message}`;
            statusMessage.textContent = 'Prediction failed.';
        } finally {
            if(predictParquetButton) predictParquetButton.disabled = false;
            // capturePredictButton is re-enabled by its own logic after capture
            if(endpoint === '/predict_parquet' && capturePredictButton) {
                 // Only re-enable capture if webcam might still be on
                capturePredictButton.disabled = !videoElement.srcObject;
            }
        }
    }

    // Clear previous results if a new file is selected or input is cleared
    if (parquetFileInput) {
        parquetFileInput.addEventListener('change', () => {
            predictionResult.textContent = '---';
            errorMessage.textContent = '';
            resultsSection.style.display = 'none';
            statusMessage.textContent = '';
        });
    }
});