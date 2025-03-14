from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_persist_directory: str = "./chroma_db"
    model_name: str = "gpt-3.5-turbo"
    
    class Config:
        env_file = ".env"

settings = Settings()
