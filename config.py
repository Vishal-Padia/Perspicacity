from typing import Set, List

class Config:
    # Constants for text processing
    MAX_CHUNK_TOKENS: int = 450
    MAX_SEQUENCE_LENGTH: int = 512
    MIN_CONTENT_LENGTH: int = 200
    
    # Search configuration
    MAX_SEARCH_ATTEMPTS: int = 10
    SEARCH_PAUSE: float = 2.0
    REQUEST_TIMEOUT: int = 10
    
    # Model configuration
    TEXT_GENERATION_MODEL: str = "google/flan-t5-base"  # Changed from large to base
    QUERY_PROCESSING_MODEL: str = "google/flan-t5-base"
    
    # Request configuration
    BLOCKED_DOMAINS: Set[str] = {
        "cloudflare.com",
        "crunchbase.com",
        "pitchbook.com",
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
    }
    
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]
    
    DEFAULT_HEADERS: dict = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
