"""Aggregated router for model registry endpoints."""
from fastapi import APIRouter
from controller.model.modelController import router as model_controller_router

model_router = APIRouter()
model_router.include_router(model_controller_router)
