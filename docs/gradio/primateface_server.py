import time
from datetime import datetime
import os
import sys
from pathlib import Path
import gradio as gr
import fastapi
import uvicorn
from fastapi.responses import JSONResponse

# Import your original gradio app
import gradio_face_detector_server as primateface_app

# Create FastAPI app FIRST
app = fastapi.FastAPI()

# Add health check to FastAPI
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "online", "timestamp": str(datetime.now())})

if __name__ == "__main__":
    print("=" * 60)
    print("PrimateFace GPU Server Starting...")
    print("=" * 60)
    
    # --- Model Pre-loading ---
    print(f"FFmpeg: {'Found' if primateface_app.check_ffmpeg() else 'Not found'}")
    print(f"MMDetection available: {primateface_app.MMDET_AVAILABLE}")
    print(f"MMPose available: {primateface_app.MMPOSE_AVAILABLE}")
    print(f"Gazelle available: {primateface_app.GAZELLE_AVAILABLE}")
    
    models_loaded_successfully = True
    
    # Critical check and pre-load for MMDetection
    if primateface_app.MMDET_AVAILABLE:
        if not Path(primateface_app.MMDET_CONFIG_PATH).exists(): 
            print(f"Warning: MMDetection config not found: {primateface_app.MMDET_CONFIG_PATH}")
        if not Path(primateface_app.MMDET_CHECKPOINT_PATH).exists(): 
            print(f"Warning: MMDetection checkpoint not found: {primateface_app.MMDET_CHECKPOINT_PATH}")
        try:
            print("Pre-loading MMDetection model...")
            primateface_app.load_mmdet_model()
            if primateface_app.mmdet_model_instance is None:
                raise RuntimeError("MMDetection model is None after loading attempt.")
            print("MMDetection model pre-loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to pre-load MMDetection model: {e}")
            models_loaded_successfully = False
    else:
        print("CRITICAL ERROR: MMDetection libraries are not available.")
        print("The application cannot function without MMDetection.")
        models_loaded_successfully = False

    # Check and pre-load for MMPose
    if models_loaded_successfully and primateface_app.MMPOSE_AVAILABLE:
        if not Path(primateface_app.MMPOSE_CONFIG_PATH).exists():
            print(f"Warning: MMPose config not found: {primateface_app.MMPOSE_CONFIG_PATH}")
        if not Path(primateface_app.MMPOSE_CHECKPOINT_PATH).exists():
            print(f"Warning: MMPose checkpoint not found: {primateface_app.MMPOSE_CHECKPOINT_PATH}")
        try:
            print("Pre-loading MMPose model...")
            primateface_app.load_mmpose_model()
            if primateface_app.mmpose_model_instance is None:
                raise RuntimeError("MMPose model failed to load.")
            print("MMPose model pre-loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to pre-load MMPose model: {e}. Face Pose Estimation task might fail.")
    elif not primateface_app.MMPOSE_AVAILABLE:
        print("Info: MMPose libraries not available. 'Face Pose Estimation' task will not function.")

    # Check and pre-load for Gazelle
    if models_loaded_successfully and primateface_app.GAZELLE_AVAILABLE:
        if not Path(primateface_app.GAZELLE_CHECKPOINT_PATH).exists():
            print(f"Warning: Gazelle checkpoint not found at {primateface_app.GAZELLE_CHECKPOINT_PATH}. 'Gaze Estimation' will fail if selected.")
        try:
            print("Pre-loading Gazelle model...")
            primateface_app.load_gazelle_model() 
            if primateface_app.gazelle_model_instance is None or primateface_app.gazelle_transform_instance is None:
                raise RuntimeError("Gazelle model or transform is None after loading attempt.")
            print("Gazelle model pre-loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to pre-load Gazelle model: {e}. 'Gaze Estimation [experimental]' task might fail if selected.")
    elif not primateface_app.GAZELLE_AVAILABLE:
        print("Info: Gazelle libraries not available. 'Gaze Estimation [experimental]' task will not function.")
        
    if models_loaded_successfully:
        # Mount Gradio at a SUB-PATH, not root, so /health remains accessible
        # app = gr.mount_gradio_app(app, primateface_app.demo, path="/gradio")
        gr.mount_gradio_app(app, primateface_app.demo, path="/gradio")
        
        print("\nServer endpoints:")
        print("  - Health check: /health")
        print("  - Gradio UI: /gradio")
        
        public_url = None
        # --- Setup ngrok tunnel ---
        try:
            from pyngrok import ngrok, conf
            
            NGROK_AUTH_TOKEN = "2zexJKab0Kkgwm1gGTzmQyYbJ7O_5f96fqyTznhWdwYB3fHgJ"
            conf.get_default().auth_token = NGROK_AUTH_TOKEN
            
            # Create tunnel
            public_url = ngrok.connect(7860, "http").public_url
            
            print("\n" + "=" * 60)
            print(f" ✓ ngrok tunnel created!")
            print(f" │")
            print(f" ├─> 🌐 PUBLIC URL: {public_url}")
            print(f" ├─> 📊 HEALTH CHECK: {public_url}/health")
            print(f" └─> 🖼️  GRADIO UI: {public_url}/gradio")
            print("=" * 60)
            
            # Save URL
            with open("primateface_backend_url.txt", "w") as f:
                f.write(str(public_url))
                
        except Exception as e:
            print(f"\n❌ Failed to create ngrok tunnel: {e}")
        
        # Launch server
        print("\nPress Ctrl+C to stop the server...")
        try:
            uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
             if public_url:
                try:
                    ngrok.disconnect(public_url)
                    print("ngrok tunnel disconnected.")
                except Exception as e:
                    print(f"Could not disconnect ngrok tunnel: {e}")

    else:
        print("\n" + "="*60)
        print("❌ FAILED TO LOAD MODELS. Please check the error messages above.")
        print("Gradio server will not start.")
        print("="*60)
        sys.exit(1) 