import os
from dotenv import load_dotenv
import streamlit as st


class AppConfig:
    def __init__(self):
        load_dotenv()

        self.knowledge_dir = "knowledge_base"
        self.model_name = "gemini-2.0-flash"
        self.embedding_model = "models/embedding-001"
        self.temperature = 0.7
        self.knowledge_dir = "./knowledge_base"  # Ou onde ficam seus PDFs/TXT
        self.vectorstore_path = "./vectorstore"  # Pasta onde o FAISS será salvo

        # tenta carregar da env local ou dos segredos do Streamlit
        self.api_key = self.get_api_key()

        if self.api_key is None:
            raise ValueError("API Key não fornecida em variável de ambiente ou st.secrets")

    def get_api_key(self):
        # 1º tenta buscar da env local (.env)
        key = os.getenv("GOOGLE_API_KEY")

        # 2º se não achou, tenta buscar do secrets (usado no Streamlit Cloud)
        if not key and "api_key" in st.secrets:
            key = st.secrets["LLM_API_KEY"]

        return key