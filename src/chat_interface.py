from typing import Dict, Any
from langchain.chains import ConversationalRetrievalChain


class ChatManager:
    def __init__(self, llm, prompt_template, memory, retriever=None):
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory = memory
        self.retriever = retriever

    def get_response(self, question: str, chat_history: list) -> Dict[str, Any]:
        """Obtém uma resposta do LLM para a pergunta do usuário"""
        system_message = {
            "role": "system",
            "content": "Você é um professor de inglês chamado Ivbezinho. Responda de forma profissional e educada."
                       "Evite cumprimentar o usuário repetidamente se já houver um cumprimento da sua parte na convesa."
        }

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": self.prompt_template,
                "document_variable_name": "context"
            }
        )

        return qa_chain.invoke({
            "question": question,
            "chat_history": [system_message] + chat_history
        })