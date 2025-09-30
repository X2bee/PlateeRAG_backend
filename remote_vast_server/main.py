from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.config_composer import config_composer
from controller.vastController import router as vastRouter
from controller.trainController import router as trainRouter
from service.database import AppDatabaseManager
from service.database.models import APPLICATION_MODELS
from service.vast.vast_service import VastService

logger = logging.getLogger("remote-vast")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for the remote Vast server."""
    logger.info("🚀 Starting remote Vast server lifespan")

    app_db = None
    try:
        database_config = config_composer.initialize_database_config_only()
        if not database_config:
            logger.error("Failed to initialize database configuration")
            yield
            return

        app_db = AppDatabaseManager(database_config)
        app_db.register_models(APPLICATION_MODELS)

        if app_db.initialize_database():
            logger.info("Application database initialized successfully")
            if app_db.run_migrations():
                logger.info("Database migrations completed successfully")
            else:
                logger.warning("Database migrations failed; continuing with caution")
            app.state.app_db = app_db
        else:
            logger.error("Application database initialization failed")
            app.state.app_db = None

        config_composer.initialize_remaining_configs()
        app.state.config_composer = config_composer

        if app.state.app_db is None:
            logger.warning("Database unavailable; Vast service will operate without persistence")

        app.state.vast_service = VastService(app.state.app_db, config_composer)
        logger.info("Vast service ready on remote server")

        yield

    finally:
        logger.info("🛑 Shutting down remote Vast server lifespan")
        # No explicit teardown required yet, placeholder for future cleanup.


def create_app() -> FastAPI:
    app = FastAPI(title="Remote Vast Server", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(vastRouter)
    app.include_router(trainRouter)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("remote_vast_server.main:app", host="0.0.0.0", port=9000, reload=False)
