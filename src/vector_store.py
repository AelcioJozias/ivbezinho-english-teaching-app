from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional
from langchain.schema import Document


class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def create_vectorstore(self, documents: list[Document]) -> Optional[FAISS]:
        """Cria e retorna um vetorstore a partir dos documentos"""
        if not documents:
            return None

        splits = self.text_splitter.split_documents(documents)
        return FAISS.from_documents(splits, self.embeddings)