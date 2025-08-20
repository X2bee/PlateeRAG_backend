# your_package/document_processor/text_handler.py
import logging, aiofiles
from typing import List
from .utils import clean_text, clean_code_text

logger = logging.getLogger("document-processor")

DEFAULT_ENCODINGS = ['utf-8','utf-8-sig','cp949','euc-kr','latin-1','ascii']

async def extract_text_from_text_file(file_path: str, file_type: str, encodings: List[str] = None, is_code: bool = False) -> str:
    enc = encodings or DEFAULT_ENCODINGS
    for e in enc:
        try:
            async with aiofiles.open(file_path, 'r', encoding=e) as f:
                t = await f.read()
            logger.info(f"Successfully read {file_path} with {e} encoding")
            return clean_code_text(t) if is_code else clean_text(t)
        except UnicodeDecodeError:
            logger.debug(f"Failed to read {file_path} with {e}, trying next...")
            continue
        except Exception as ex:
            logger.error(f"Error reading file {file_path} with {e}: {ex}")
            continue
    raise Exception(f"Could not read file {file_path} with any supported encoding")
