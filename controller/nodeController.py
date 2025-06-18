from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import threading
import logging
import os
import signal
import subprocess
import json
from datetime import datetime
import glob
from pathlib import Path


router = APIRouter(
    prefix="/node",
    tags=["node"],
    responses={404: {"description": "Not found"}},
)