from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional
from langchain.schema import Document
import os

class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore_path = self.config.vectorstore_path  # Ex: "./vectorstore"

    def create_vectorstore(self, documents: list[Document]) -> Optional[FAISS]:
        """Cria e salva o vectorstore a partir dos documentos"""
        if not documents:
            return None

        splits = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Cria pasta se não existir
        os.makedirs(self.vectorstore_path, exist_ok=True)

        # Salva FAISS localmente
        vectorstore.save_local(self.vectorstore_path)

        return vectorstore

    def load_vectorstore(self) -> Optional[FAISS]:
        """Tenta carregar o vectorstore do disco, se existir"""
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        if os.path.exists(index_path):
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None
