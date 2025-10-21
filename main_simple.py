from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI()

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

@app.get("/api/admin/base/superuser")
async def get_superuser():
    return {"exists": False, "message": "No superuser found"}

@app.post("/api/admin/base/create-superuser")
async def create_superuser(request: Request):
    try:
        body = await request.body()
        print(f"Raw body: {body}")
        
        if body:
            data = await request.json()
            print(f"Parsed JSON: {data}")
        
        return {"success": True, "message": "Superuser created successfully"}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)