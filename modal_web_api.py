"""
Modal Web API for CourtCheck - Frontend Integration.

This provides web endpoints for the frontend to upload videos and get processing results.
"""

import modal
import os
import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json

# Import the image and volumes from modal_deploy
from modal_deploy import image, videos_volume, app as main_app

# Create FastAPI app
web_app = FastAPI()

# Add CORS middleware for frontend
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a separate app for web endpoints
app = modal.App("courtcheck-web")


@app.function(
    image=image,
    volumes={"/videos": videos_volume},
)
@modal.asgi_app()
def fastapi_app():
    """Mount FastAPI app as Modal ASGI endpoint."""
    
    @web_app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        """
        Upload a video file.
        
        Returns:
            video_id: Unique identifier for the uploaded video
        """
        try:
            # Generate unique video ID
            import uuid
            video_id = str(uuid.uuid4())
            
            # Save video to Modal volume
            video_path = f"/videos/{video_id}_{file.filename}"
            
            contents = await file.read()
            with open(video_path, "wb") as f:
                f.write(contents)
            
            # Commit volume changes
            videos_volume.commit()
            
            return JSONResponse({
                "video_id": video_id,
                "filename": file.filename,
                "size": len(contents),
                "status": "uploaded",
                "message": "Video uploaded successfully"
            })
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @web_app.post("/api/process/{video_id}")
    async def process_video_api(video_id: str, filename: str):
        """
        Process an uploaded video.
        
        Args:
            video_id: Unique identifier for the video
            filename: Original filename
            
        Returns:
            job_id: Identifier for tracking processing status
        """
        try:
            video_path = f"/videos/{video_id}_{filename}"
            output_path = f"/videos/{video_id}_output.mp4"
            
            # Check if video exists
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Spawn processing job (async)
            from modal_deploy import process_video
            process_video.spawn(video_path=video_path, output_path=output_path)
            
            return JSONResponse({
                "job_id": video_id,
                "status": "processing",
                "message": "Video processing started"
            })
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @web_app.get("/api/status/{video_id}")
    async def get_status(video_id: str):
        """
        Get processing status for a video.
        
        Args:
            video_id: Unique identifier for the video
            
        Returns:
            status: current processing status
        """
        try:
            output_path = f"/videos/{video_id}_output.mp4"
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return JSONResponse({
                    "video_id": video_id,
                    "status": "completed",
                    "output_size": file_size,
                    "download_url": f"/api/download/{video_id}"
                })
            else:
                return JSONResponse({
                    "video_id": video_id,
                    "status": "processing",
                    "message": "Video is being processed"
                })
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @web_app.get("/api/download/{video_id}")
    async def download_video(video_id: str):
        """
        Download processed video.
        
        Args:
            video_id: Unique identifier for the video
            
        Returns:
            StreamingResponse with video file
        """
        try:
            output_path = f"/videos/{video_id}_output.mp4"
            
            if not os.path.exists(output_path):
                raise HTTPException(status_code=404, detail="Processed video not found")
            
            def iterfile():
                with open(output_path, "rb") as f:
                    yield from f
            
            return StreamingResponse(
                iterfile(),
                media_type="video/mp4",
                headers={"Content-Disposition": f"attachment; filename={video_id}_output.mp4"}
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @web_app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "CourtCheck API"}
    
    return web_app


if __name__ == "__main__":
    # For local testing
    print("Deploy with: modal deploy modal_web_api.py")
    print("After deployment, your API will be available at the Modal URL")
