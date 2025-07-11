"""
이메일 알림 관련 설정
"""
from typing import Dict
from config.base_config import BaseConfig, PersistentConfig

class EmailConfig(BaseConfig):
    """이메일 알림 관련 설정 관리"""

    def initialize(self) -> Dict[str, PersistentConfig]:
        """이메일 관련 설정들을 초기화"""

        # SMTP 서버 설정
        self.SMTP_HOST = self.create_persistent_config(
            env_name="EMAIL_SMTP_HOST",
            config_path="email.smtp.host",
            default_value="smtp.gmail.com"
        )

        self.SMTP_PORT = self.create_persistent_config(
            env_name="EMAIL_SMTP_PORT",
            config_path="email.smtp.port",
            default_value=587
        )

        self.SMTP_USERNAME = self.create_persistent_config(
            env_name="EMAIL_SMTP_USERNAME",
            config_path="email.smtp.username",
            default_value="Unset",
        )

        self.SMTP_PASSWORD = self.create_persistent_config(
            env_name="EMAIL_SMTP_PASSWORD",
            config_path="email.smtp.password",
            default_value="Unset",
            file_path="email_password.txt"
        )

        # 이메일 알림 설정
        self.NOTIFICATIONS_ENABLED = self.create_persistent_config(
            env_name="EMAIL_NOTIFICATIONS_ENABLED",
            config_path="email.notifications.enabled",
            default_value=False
        )

        self.ADMIN_EMAIL = self.create_persistent_config(
            env_name="EMAIL_ADMIN_EMAIL",
            config_path="email.admin.email",
            default_value="admin@example.com"
        )

        self.FROM_EMAIL = self.create_persistent_config(
            env_name="EMAIL_FROM_EMAIL",
            config_path="email.from.email",
            default_value="noreply@plateerag.com"
        )

        # 반환할 설정들
        return {
            "EMAIL_SMTP_HOST": self.SMTP_HOST,
            "EMAIL_SMTP_PORT": self.SMTP_PORT,
            "EMAIL_SMTP_USERNAME": self.SMTP_USERNAME,
            "EMAIL_SMTP_PASSWORD": self.SMTP_PASSWORD,
            "EMAIL_NOTIFICATIONS_ENABLED": self.NOTIFICATIONS_ENABLED,
            "EMAIL_ADMIN_EMAIL": self.ADMIN_EMAIL,
            "EMAIL_FROM_EMAIL": self.FROM_EMAIL
        }
