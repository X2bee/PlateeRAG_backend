from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os

# Create FastAPI app
app = FastAPI(
    title="PlateeRAG Backend API",
    description="Minimal API for testing",
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

@app.get("/")
async def root():
    return {"message": "PlateeRAG Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "plateerag-backend"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)