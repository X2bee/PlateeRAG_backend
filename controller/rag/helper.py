from fastapi import Request
import logging
import gc

logger = logging.getLogger("embedding-controller")

def safely_replace_embedding_client(request: Request, new_embedding_client):
    try:
        if hasattr(request.app.state, 'embedding_client') and request.app.state.embedding_client is not None:
            old_client = request.app.state.embedding_client
            logger.info("Cleaning up existing embedding client")

            if hasattr(old_client, 'cleanup'):
                try:
                    old_client.cleanup()
                except Exception as e:
                    logger.warning(f"Error during embedding client cleanup: {e}")

            if hasattr(old_client, 'close'):
                try:
                    old_client.close()
                except Exception as e:
                    logger.warning(f"Error during embedding client close: {e}")

            request.app.state.embedding_client = None
            del old_client

            gc.collect()
            logger.info("Old embedding client cleaned up successfully")

        request.app.state.embedding_client = new_embedding_client
        logger.info("New embedding client assigned successfully")

    except Exception as e:
        logger.error(f"Error during embedding client replacement: {e}")
        request.app.state.embedding_client = new_embedding_client
        raise
