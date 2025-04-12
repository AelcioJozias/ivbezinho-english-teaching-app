import streamlit as st
from src.config import AppConfig
from src.document_loader import DocumentLoader
from src.memory_manager import MemoryManager
from src.llm_setup import LLMSetup
from src.vector_store import VectorStoreManager
from src.chat_interface import ChatManager


def initialize_session():
    """Inicializa todos os componentes da sessão"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager()

    if "vectorstore" not in st.session_state:
        config = AppConfig()
        loader = DocumentLoader(config.knowledge_dir)
        documents = loader.load_documents()

        vector_manager = VectorStoreManager(config)
        st.session_state.vectorstore = vector_manager.create_vectorstore(documents)

        llm_setup = LLMSetup(config)
        st.session_state.llm = llm_setup.get_llm()
        st.session_state.prompt_template = llm_setup.get_prompt_template()


def clear_conversation():
    """Limpa a conversa e reinicia a aplicação"""
    st.session_state.chat_history = []
    st.session_state.memory.clear_memory()
    st.rerun()


def main():
    st.set_page_config(page_title="Ivbezinho - Seu Professor de Inglês", page_icon="🇬🇧", layout="wide")
    st.title("🇬🇧 Ivbezinho - Seu Professor de Inglês")
    st.markdown("Bem-vindo ao Ivbezinho, seu assistente de ensino de inglês!")

    initialize_session()

    # Sidebar for configurations and clear button
    with st.sidebar:
        st.header("Configurações")
        if st.button("Limpar Conversa"):
            clear_conversation()

    # Main chat interface
    st.container()  # Placeholder for future layout adjustments
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user prompt
    prompt = st.chat_input("Digite sua pergunta sobre inglês")

    # Process user input
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_manager = ChatManager(
            llm=st.session_state.llm,
            prompt_template=st.session_state.prompt_template,
            memory=st.session_state.memory.memory,
            retriever=st.session_state.vectorstore.as_retriever() if st.session_state.vectorstore else None
        )

        with st.spinner("Ivbezinho está pensando..."):
            try:
                response = chat_manager.get_response(prompt, st.session_state.chat_history)
                answer = response["answer"]
            except Exception as e:
                answer = f"Desculpe, ocorreu um erro: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()

