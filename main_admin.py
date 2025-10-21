from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import json

# Create FastAPI app
app = FastAPI(
    title="PlateeRAG Backend API",
    description="Backend API with Admin functionality",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug middleware
@app.middleware("http")
async def debug_requests(request: Request, call_next):
    print(f"Request: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "PlateeRAG Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "plateerag-backend"}

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API proxy is working!", "timestamp": "2025-10-21"}

# Request models
class CreateSuperuserRequest(BaseModel):
    username: str
    password: str
    email: str = None
    full_name: str = None

# Basic admin endpoints for testing
@app.get("/api/admin/base/superuser")
async def get_superuser():
    return {"exists": False, "message": "No superuser found"}

@app.post("/api/admin/base/create-superuser")
async def create_superuser(request: Request):
    try:
        # Get raw body for debugging
        body = await request.body()
        print(f"Raw request body: {body}")
        
        # Parse JSON
        data = await request.json()
        print(f"Parsed JSON data: {data}")
        
        username = data.get("username", "unknown")
        print(f"Creating superuser: {username}")
        
        # Simulate successful creation
        return {
            "success": True, 
            "message": "Superuser created successfully",
            "user": {
                "username": username,
                "email": data.get("email"),
                "full_name": data.get("full_name")
            }
        }
    except Exception as e:
        print(f"Error creating superuser: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)