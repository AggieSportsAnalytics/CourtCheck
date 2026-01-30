"""
Local Development Runner for CourtCheck
Starts both backend and frontend servers
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("[CHECK] Checking dependencies...")
    
    # Check Python packages
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        print("[OK] Python dependencies OK")
    except ImportError as e:
        print(f"[ERROR] Missing Python package: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check Node.js
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        print("[OK] Node.js OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Node.js not found")
        print("Install from: https://nodejs.org/")
        return False
    
    # Check frontend dependencies
    if not (Path("frontend/node_modules").exists()):
        print("[WARNING] Frontend dependencies not installed")
        print("Installing...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd="frontend",
                check=True
            )
            print("[OK] Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install frontend dependencies")
            return False
    else:
        print("[OK] Frontend dependencies OK")
    
    return True

def check_models():
    """Check if model weights are present."""
    print("\n[CHECK] Checking model weights...")
    
    models = {
        "TrackNet": "tracknet_weights.pt",
        "Court Detection": "model_tennis_court_det.pt",
        "Stroke Classifier": "stroke_classifier_weights.pth",
        "Bounce Detection": "bounce_detection_weights.cbm",
        "COCO Results": "coco_instances_results.json"
    }
    
    all_present = True
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[OK] {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"[WARNING] {name}: {path} (NOT FOUND)")
            all_present = False
    
    if not all_present:
        print("\n[WARNING] Some models are missing. The app will still run but with reduced functionality.")
    
    return True  # Continue even if some models are missing

def start_backend():
    """Start the backend server."""
    print("\n[START] Starting backend server...")
    print("[URL] Backend will be available at: http://localhost:8000")
    print("[DOCS] API docs at: http://localhost:8000/docs")
    
    backend_process = subprocess.Popen(
        [sys.executable, "local_backend.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for backend to start
    print("[WAIT] Waiting for backend to start...")
    time.sleep(3)
    
    return backend_process

def start_frontend():
    """Start the frontend server."""
    print("\n[START] Starting frontend server...")
    print("[URL] Frontend will be available at: http://localhost:3000")
    
    # Use shell=True on Windows to find npm in PATH
    frontend_process = subprocess.Popen(
        "npm start",
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        shell=True
    )
    
    return frontend_process

def main():
    """Main entry point."""
    print("=" * 60)
    print("COURTCHECK - LOCAL DEVELOPMENT SERVER")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Dependency check failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Check models
    check_models()
    
    # Start servers
    backend_process = start_backend()
    frontend_process = start_frontend()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] CourtCheck is running!")
    print("=" * 60)
    print("\nAccess Points:")
    print("   - Frontend:  http://localhost:3000")
    print("   - Backend:   http://localhost:8000")
    print("   - API Docs:  http://localhost:8000/docs")
    print("\nTips:")
    print("   - Upload a tennis video through the web interface")
    print("   - Check backend logs for processing status")
    print("   - Processed videos are saved in ./outputs/")
    print("\nPress CTRL+C to stop both servers\n")
    
    try:
        # Keep running and display logs
        while True:
            # Read backend output
            if backend_process.poll() is None:
                line = backend_process.stdout.readline()
                if line:
                    print(f"[BACKEND] {line.rstrip()}")
            else:
                print("[ERROR] Backend stopped unexpectedly")
                break
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for clean shutdown
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)
        
        print("[SUCCESS] Servers stopped")
        print("Goodbye!")

if __name__ == "__main__":
    main()
