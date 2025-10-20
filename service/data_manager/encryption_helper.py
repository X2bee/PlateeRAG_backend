"""
암호화 헬퍼 - 민감 정보 보호
"""
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class EncryptionHelper:
    """비밀번호 암호화/복호화"""
    
    def __init__(self, secret_key: str = None):
        if secret_key is None:
            secret_key = os.environ.get('ENCRYPTION_SECRET_KEY', 'your-secret-key-change-in-production')
        
        # 키 파생
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'data_manager_salt_v1',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """암호화"""
        if not plaintext:
            return ""
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """복호화"""
        if not ciphertext:
            return ""
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

# 싱글톤 인스턴스
_encryption_helper = None

def get_encryption_helper() -> EncryptionHelper:
    """싱글톤 인스턴스 반환"""
    global _encryption_helper
    if _encryption_helper is None:
        _encryption_helper = EncryptionHelper()
    return _encryption_helper