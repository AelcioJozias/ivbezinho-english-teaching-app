from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import List
from langchain.schema import Document
import os


class DocumentLoader:
    _cached_documents: List[Document] = None  # cache em nível de classe

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir

    def load_documents(self) -> List[Document]:
        """Carrega documentos da pasta de conhecimento apenas uma vez"""
        if DocumentLoader._cached_documents is not None:
            return DocumentLoader._cached_documents  # retorna do cache

        documents = []

        if not os.path.exists(self.knowledge_dir):
            return documents

        for file in os.listdir(self.knowledge_dir):
            file_path = os.path.join(self.knowledge_dir, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Erro ao carregar o arquivo {file}: {e}")

        # Salva no cache
        DocumentLoader._cached_documents = documents
        return documents
