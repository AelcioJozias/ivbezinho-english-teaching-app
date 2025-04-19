import streamlit as st
from src.config import AppConfig
from src.document_loader import DocumentLoader
from src.memory_manager import MemoryManager
from src.llm_setup import LLMSetup
from src.vector_store import VectorStoreManager
from src.chat_interface import ChatManager

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Ivbezinho - Seu Professor de Inglês", page_icon="ivb", layout="centered")

def initialize_session():
    """Inicializa todos os componentes da sessão"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager()

    if "vectorstore" not in st.session_state:
        config = AppConfig()
        vector_manager = VectorStoreManager(config)

        # Primeiro tenta carregar o vectorstore do disco
        vectorstore = vector_manager.load_vectorstore()

        # Se não encontrar, carrega os documentos e cria o vectorstore
        if vectorstore is None:
            loader = DocumentLoader(config.knowledge_dir)
            documents = loader.load_documents()
            vectorstore = vector_manager.create_vectorstore(documents)

        st.session_state.vectorstore = vectorstore

    if "llm" not in st.session_state or "prompt_template" not in st.session_state:
        llm_setup = LLMSetup(config)
        st.session_state.llm = llm_setup.get_llm()
        st.session_state.prompt_template = llm_setup.get_prompt_template()


def clear_conversation():
    """Limpa a conversa e reinicia a aplicação"""
    st.session_state.chat_history = []
    st.session_state.memory.clear_memory()
    st.rerun()


def apply_custom_styles():
    """Aplica estilos personalizados para o frontend."""
    st.markdown(
        """
        <style> 
# #MainMenu {visibility: hidden;} 
# footer {visibility: hidden;} 
# header {visibility: hidden;} 
</style>
        """,
        unsafe_allow_html=True
    )


def main():
    apply_custom_styles()  # Chama a função para aplicar os estilos
    col1, col2 = st.columns([1, 5], vertical_alignment="center")
    with col1:
        st.image("image/Ivbezinho - tamanho de arquivo reduzido.png", width=100)  # Ajuste o tamanho conforme necessário
    with col2:
        st.title("Ivebezinho")

    st.markdown(
        '<p style="font-size:20px;">Hello! Sou Ivebezinho! O tutor da academia IVB. Em que posso te ajudar?</p>',
        unsafe_allow_html=True)

    initialize_session()

    # Sidebar for configurations and clear button
    with st.sidebar:
        st.image("image/logo.jpeg", width=200)
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

