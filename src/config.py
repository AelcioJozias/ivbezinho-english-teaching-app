import os
from dotenv import load_dotenv


class AppConfig:
    def __init__(self):
        load_dotenv()
        self.knowledge_dir = "knowledge_base"
        self.model_name = "gemini-1.5-flash"
        self.embedding_model = "models/embedding-001"
        self.temperature = 0.7

    @property
    def google_api_key(self):
        return os.getenv("GOOGLE_API_KEY")