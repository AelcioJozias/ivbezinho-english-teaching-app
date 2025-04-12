from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage


class MemoryManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def clear_memory(self):
        """Limpa a memória da conversa"""
        self.memory.clear()

    def get_chat_history(self) -> list[BaseMessage]:
        """Retorna o histórico da conversa"""
        return self.memory.chat_memory.messages