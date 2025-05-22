document.addEventListener('DOMContentLoaded', () => {
    // Mode switching using Nav buttons
    const uploadModeBtnNav = document.getElementById('uploadModeBtnNav');
    const webcamModeBtnNav = document.getElementById('webcamModeBtnNav');
    const uploadSection = document.getElementById('uploadSection');
    const webcamSection = document.getElementById('webcamSection');

    function switchMode(activeBtn, inactiveBtn, activeContent, inactiveContent) {
        activeContent.classList.add('active-content');
        inactiveContent.classList.remove('active-content');
        activeBtn.classList.add('active');
        inactiveBtn.classList.remove('active');
    }

    uploadModeBtnNav.addEventListener('click', () => {
        switchMode(uploadModeBtnNav, webcamModeBtnNav, uploadSection, webcamSection);
        stopWebcam(); 
    });

    webcamModeBtnNav.addEventListener('click', () => {
        switchMode(webcamModeBtnNav, uploadModeBtnNav, webcamSection, uploadSection);
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
    const playSpeechButton = document.getElementById('playSpeechButton');
    const audioPlayback = document.getElementById('audioPlayback');


    let hands, pose, camera;
    let landmarkFrames = [];
    let isCapturing = false;
    const CAPTURE_DURATION_MS = 3000; 
    const CAPTURE_FPS = 20; 
    const CAPTURE_FRAMES_COUNT = CAPTURE_DURATION_MS / 1000 * CAPTURE_FPS;
    let captureIntervalId = null;

    const LPOSE_INDICES = [13, 15, 17, 19, 21];
    const RPOSE_INDICES = [14, 16, 18, 20, 22];
    const POSE_LANDMARK_INDICES = [...LPOSE_INDICES, ...RPOSE_INDICES]; 
    const NUM_HAND_LANDMARKS = 21;
    const NUM_POSE_LANDMARKS_TO_EXTRACT = POSE_LANDMARK_INDICES.length; 
    const TOTAL_FEATURES_PER_COORD = NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS_TO_EXTRACT; 
    const TOTAL_FEATURES = TOTAL_FEATURES_PER_COORD * 3; 

    function onResults(results) {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        const frameLandmarks = new Array(TOTAL_FEATURES).fill(NaN);

        if (results.poseLandmarks) {
            drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
            drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
            
            POSE_LANDMARK_INDICES.forEach((poseIdx, i) => {
                if (results.poseLandmarks[poseIdx]) {
                    frameLandmarks[NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].x; 
                    frameLandmarks[TOTAL_FEATURES_PER_COORD + NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].y; 
                    frameLandmarks[TOTAL_FEATURES_PER_COORD * 2 + NUM_HAND_LANDMARKS * 2 + i] = results.poseLandmarks[poseIdx].z; 
                }
            });
        }

        if (results.multiHandLandmarks) {
            for (let handIndex = 0; handIndex < results.multiHandLandmarks.length; handIndex++) {
                const landmarks = results.multiHandLandmarks[handIndex];
                const classification = results.multiHandedness[handIndex];
                const isRightHand = classification.label === 'Right';

                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: isRightHand ? '#00FF00' : '#FF0000', lineWidth: 3 });
                drawLandmarks(canvasCtx, landmarks, { color: isRightHand ? '#FFFFFF' : '#0000FF', lineWidth: 1, radius: 3 });

                const offset = isRightHand ? 0 : NUM_HAND_LANDMARKS;
                landmarks.forEach((landmark, i) => {
                    frameLandmarks[offset + i] = landmark.x; 
                    frameLandmarks[TOTAL_FEATURES_PER_COORD + offset + i] = landmark.y; 
                    frameLandmarks[TOTAL_FEATURES_PER_COORD * 2 + offset + i] = landmark.z; 
                });
            }
        }
        canvasCtx.restore();

        if (isCapturing && landmarkFrames.length < CAPTURE_FRAMES_COUNT) {
            landmarkFrames.push({ landmarks: frameLandmarks });
        }
    }

    if (typeof Hands !== 'undefined' && typeof Pose !== 'undefined') {
        hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        hands.onResults(onResults);

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
    } else {
        console.error("MediaPipe Hands or Pose class not found. Check CDN links.");
        statusMessage.textContent = "Error loading MediaPipe libraries.";
    }
    
    async function processFrame() {
        if (videoElement && !videoElement.paused && !videoElement.ended) {
            if (hands) await hands.send({ image: videoElement });
            if (pose) await pose.send({ image: videoElement });
        }
        if (videoElement.srcObject) { 
            requestAnimationFrame(processFrame);
        }
    }
    
    if (predictParquetButton) {
        predictParquetButton.addEventListener('click', async () => {
            const file = parquetFileInput.files[0];
            if (!file) {
                errorMessage.textContent = 'Please select a Parquet file.';
                resetResults();
                return;
            }
            await performPrediction(file, '/predict_parquet');
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
                    if (typeof Camera !== 'undefined') {
                        camera = new Camera(videoElement, {
                            onFrame: async () => {},
                            width: 640,
                            height: 480
                        });
                        camera.start(); 
                        processFrame(); 
                        statusMessage.textContent = 'Webcam started. Position your hands.';
                    } else {
                        throw new Error("MediaPipe Camera utility not found.");
                    }
                };
                startWebcamButton.disabled = true;
                stopWebcamButton.disabled = false;
                capturePredictButton.disabled = false;
                errorMessage.textContent = '';
                resetResults();
            } catch (err) {
                console.error("Error accessing webcam:", err);
                errorMessage.textContent = 'Error accessing webcam. Please ensure permissions are granted and MediaPipe libraries are loaded.';
                statusMessage.textContent = '';
            }
        });
    }
    
    function stopWebcam() {
        if (camera) {
            camera.stop(); 
            camera = null;
        }
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
        if(captureIntervalId) clearInterval(captureIntervalId);
        captureIntervalId = null;
        isCapturing = false;
        if(startWebcamButton) startWebcamButton.disabled = false;
        if(stopWebcamButton) stopWebcamButton.disabled = true;
        if(capturePredictButton) capturePredictButton.disabled = true;
        statusMessage.textContent = 'Webcam stopped.';
        if(canvasCtx) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
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
            capturePredictButton.disabled = true; 
            resetResults();

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
                if(capturePredictButton) capturePredictButton.disabled = !videoElement.srcObject;
            }, CAPTURE_DURATION_MS);
        });
    }

    async function performPrediction(payload, endpoint) {
        predictionResult.textContent = 'Processing...';
        resultsSection.style.display = 'block';
        errorMessage.textContent = '';
        playSpeechButton.style.display = 'none';
        audioPlayback.src = ''; // Clear previous audio
        
        if(predictParquetButton) predictParquetButton.disabled = true;
        if(capturePredictButton && endpoint === '/predict_live_data') { // Only disable capture button for its own action
            capturePredictButton.disabled = true;
        }


        let requestOptions;
        if (payload instanceof File) { 
            const formData = new FormData();
            formData.append('file', payload);
            requestOptions = { method: 'POST', body: formData };
        } else { 
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

            if (data.audio_base64) {
                audioPlayback.src = `data:audio/mp3;base64,${data.audio_base64}`;
                playSpeechButton.style.display = 'inline-block'; // Show the play button
            } else {
                playSpeechButton.style.display = 'none';
                if (data.prediction) { // If there's a prediction but no audio
                    statusMessage.textContent += ' (TTS disabled or failed)';
                }
            }

        } catch (error) {
            console.error('Error:', error);
            resetResults();
            errorMessage.textContent = `Error: ${error.message}`;
            statusMessage.textContent = 'Prediction failed.';
        } finally {
            if(predictParquetButton) predictParquetButton.disabled = false;
            if(capturePredictButton && endpoint === '/predict_live_data') { // Only re-enable if it was its action
                 capturePredictButton.disabled = !videoElement.srcObject; 
            } else if (capturePredictButton && endpoint !== '/predict_live_data'){
                // If parquet prediction, ensure capture button state reflects webcam state
                capturePredictButton.disabled = !videoElement.srcObject;
            }
        }
    }
    
    function resetResults() {
        predictionResult.textContent = '---';
        resultsSection.style.display = 'none';
        playSpeechButton.style.display = 'none';
        audioPlayback.src = '';
    }
    
    if (playSpeechButton) {
        playSpeechButton.addEventListener('click', () => {
            if (audioPlayback.src && audioPlayback.src !== document.location.href + "#") { 
                audioPlayback.play().catch(e => {
                    console.error("Error playing audio:", e);
                    errorMessage.textContent = "Error playing audio. Check console.";
                });
            } else {
                errorMessage.textContent = "No audio to play for the current prediction.";
            }
        });
    }

    if (parquetFileInput) {
        parquetFileInput.addEventListener('change', () => {
            resetResults();
            errorMessage.textContent = '';
            statusMessage.textContent = '';
        });
    }
    
    // Initialize with upload mode active
    switchMode(uploadModeBtnNav, webcamModeBtnNav, uploadSection, webcamSection);
});