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
logger = logging.getLogger("polar-trainer")

app = FastAPI(
    title="PlateeRAG Backend",
    description="API for training models with customizable parameters",
    version="1.0.0"
)

if "OPENAI_API_KEY" not in os.environ:
    with open('openai_api_key.txt', 'r') as api:
        os.environ["OPENAI_API_KEY"] = api.read()

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
    logger.info("Starting node discovery...")
    run_discovery()
    generate_json_spec("constants/exported_nodes.json")
    logger.info("Node discovery completed successfully!")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)