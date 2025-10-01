"""
Token counting utilities using tiktoken
"""
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("token-counter")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Token counting will use approximate character-based method.")

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken if available, otherwise use approximation.
    
    Args:
        text: Text to count tokens for
        model: Model name to determine encoding (default: gpt-3.5-turbo)
    
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    if not isinstance(text, str):
        text = str(text)
    
    if TIKTOKEN_AVAILABLE:
        try:
            # Map model names to encodings
            encoding_map = {
                "gpt-3.5-turbo": "cl100k_base",
                "gpt-4": "cl100k_base", 
                "gpt-4o": "o200k_base",
                "text-davinci-003": "p50k_base",
                "text-davinci-002": "p50k_base",
                "code-davinci-002": "p50k_base",
            }
            
            encoding_name = encoding_map.get(model, "cl100k_base")
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
            
        except Exception as e:
            logger.warning(f"Failed to count tokens with tiktoken: {e}. Using approximation.")
            return approximate_token_count(text)
    else:
        return approximate_token_count(text)

def approximate_token_count(text: str) -> int:
    """
    Approximate token count using character-based estimation.
    Rough estimate: 1 token ≈ 4 characters for English text.
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Approximate number of tokens
    """
    if not text:
        return 0
    
    # Remove extra whitespace and count characters
    cleaned_text = ' '.join(text.split())
    char_count = len(cleaned_text)
    
    # Rough approximation: 1 token ≈ 4 characters
    # Adjust for different languages and content types
    if any(ord(char) > 127 for char in cleaned_text):  # Contains non-ASCII (likely CJK)
        # CJK characters are typically 1 token per character
        return int(char_count * 0.8)  # Slightly less due to mixed content
    else:
        # English/Latin text
        return max(1, char_count // 4)

def extract_and_count_tokens(data: Any, model: str = "gpt-3.5-turbo") -> int:
    """
    Extract text from various data formats and count tokens.
    
    Args:
        data: Data to extract text from (str, dict, list, etc.)
        model: Model name for token counting
    
    Returns:
        Number of tokens
    """
    if not data:
        return 0
    
    try:
        # If it's already a string, count directly
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(data)
                return extract_and_count_tokens(parsed_data, model)
            except (json.JSONDecodeError, TypeError):
                # Not JSON, count as plain text
                return get_token_count(data, model)
        
        # If it's a dictionary, extract relevant text fields
        elif isinstance(data, dict):
            text_parts = []
            
            # Common text fields to extract
            text_fields = ['text', 'content', 'message', 'input', 'output', 'result', 'response']
            
            for field in text_fields:
                if field in data and data[field]:
                    text_parts.append(str(data[field]))
            
            # If no common fields found, convert entire dict to string
            if not text_parts:
                text_parts.append(json.dumps(data, ensure_ascii=False))
            
            combined_text = ' '.join(text_parts)
            return get_token_count(combined_text, model)
        
        # If it's a list, process each item
        elif isinstance(data, list):
            total_tokens = 0
            for item in data:
                total_tokens += extract_and_count_tokens(item, model)
            return total_tokens
        
        # For other types, convert to string
        else:
            return get_token_count(str(data), model)
    
    except Exception as e:
        logger.error(f"Error counting tokens for data: {e}")
        # Fallback to string conversion
        return get_token_count(str(data), model)

def count_io_tokens(input_data: Any, output_data: Any, model: str = "gpt-3.5-turbo") -> Dict[str, int]:
    """
    Count tokens for input and output data.
    
    Args:
        input_data: Input data to count tokens for
        output_data: Output data to count tokens for
        model: Model name for token counting
    
    Returns:
        Dictionary with input_tokens, output_tokens, and total_tokens
    """
    input_tokens = extract_and_count_tokens(input_data, model)
    output_tokens = extract_and_count_tokens(output_data, model)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }