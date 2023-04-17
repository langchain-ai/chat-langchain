import os

from loguru import logger
from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index: str
    docchat_database_url: str
    websocket_endpoint: str


def get_project_root() -> str:
    return os.path.join(os.path.dirname(__file__), "../../")


def get_settings() -> Settings:
    settings = initialize_from_env_file(".ENV")
    return settings


def initialize_from_env_file(env_file_name: str) -> Settings:
    env_file_path = os.path.join(get_project_root(), env_file_name)
    logger.info(f"Settings:initialize_from_env_file loading from: {env_file_path}")
    settings = Settings(_env_file=env_file_path, _env_file_encoding="utf-8")
    return settings
