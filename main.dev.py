from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from controller.nodeController import router as nodeRouter
from controller.configController import router as configRouter
from controller.workflowController import router as workflowRouter
from src.node_composer import run_discovery, generate_json_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("polar-trainer-dev")

def ensure_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 없으면 생성"""
    directories = ["constants", "downloads"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created {directory} directory")
        else:
            logger.info(f"{directory} directory already exists")

app = FastAPI(
    title="PlateeRAG Backend (Development)",
    description="API for training models with customizable parameters - Development Server",
    version="1.0.0-dev"
)

if "OPENAI_API_KEY" not in os.environ:
    try:
        with open('openai_api_key.txt', 'r') as api:
            os.environ["OPENAI_API_KEY"] = api.read()
    except FileNotFoundError:
        logger.warning("openai_api_key.txt file not found. Please set OPENAI_API_KEY environment variable.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(nodeRouter)
app.include_router(configRouter)
app.include_router(workflowRouter)

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 노드 discovery 실행"""
    try:
        logger.info("Starting node discovery...")
        
        # 필요한 디렉토리들 생성
        ensure_directories()
        
        run_discovery()
        generate_json_spec("constants/exported_nodes.json")
        logger.info("Node discovery completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.info("Application will continue despite startup error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
