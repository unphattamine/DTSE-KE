import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

##
#streamlit cloud running config to use tmp as cache
#os.environ['LLAMA_INDEX_CACHE_DIR'] = '/tmp/llama_index_cache'
#import nltk
#nltk_data_path = "/tmp/nltk_data"
#if not os.path.exists(nltk_data_path):
#    os.makedirs(nltk_data_path, exist_ok=True)
#nltk.data.path.insert(0, nltk_data_path)
#try:
#    nltk.data.find('tokenizers/punkt')
#except LookupError:
#    nltk.download('punkt', download_dir=nltk_data_path)
#    nltk.download('punkt_tab', download_dir=nltk_data_path)
##

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter

st.set_page_config(page_title="Deutsche Telekom Press release RAG", layout="wide")

load_dotenv()
api_key = os.getenv("API_KEY")

#llm settings
st.sidebar.title("Settings")
llm_model_options = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "Kimi K2 Instruct": "moonshotai/kimi-k2-instruct-0905"
}
emb_model_options = {
    "bge-small-en-v1.5":"BAAI/bge-small-en-v1.5"
}

selected_llm_model = st.sidebar.selectbox(
    "LLM Model:",
    options=list(llm_model_options.keys()),
    index=0
)
selected_emb_model = st.sidebar.selectbox(
    "Emb Model:",
    options=list(emb_model_options.keys()),
    index=0
)
selected_llm_model_id = llm_model_options[selected_llm_model]
selected_emb_model_id = emb_model_options[selected_emb_model]

Settings.llm = Groq(model=selected_llm_model_id, api_key=api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name=selected_emb_model_id)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

#db files check
def check_files():
    if not os.path.exists("./data") or not os.listdir("./data"):
        st.error("No files found in /data folder")
        st.stop()

def count_all_files():
    check_files()
    path = Path("./data")
    file_count = sum(1 for item in path.iterdir() if item.is_file())
    return file_count

st.sidebar.divider()
st.sidebar.info(f"**Active Model:** {selected_llm_model}")
st.sidebar.info(f"**Embedding modeL:**  {selected_emb_model}")
st.sidebar.info(f"**Data:** {count_all_files()} files in ./data")


if st.sidebar.button("Restart"):
    st.session_state.messages = []
    st.rerun()

@st.cache_resource(show_spinner="Analyzing dataset")

def get_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #if db exists
    if chroma_collection.count() == 0:
        check_files()

        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index

index = get_index()

#chat engine
chat_engine = index.as_chat_engine(
    chat_mode="context",
    similarity_top_k=3,
    system_prompt=(
        "You are a press assistant. Answer only on provided context. "
        "If you dont have the information, say sorry, i dont know, but ask me tomorrow"
    )
)

st.title("Deutsche Telekom Press release RAG")
if "messages" not in st.session_state:
    st.session_state.messages = []

#chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about recent press releases"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #gen response
    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt)
        st.markdown(response.response)

        #source
        with st.expander("Sources"):
            for i, node in enumerate(response.source_nodes):
                st.markdown(f"**Source {i + 1} (Match Score: {round(node.score, 3)}):**")
                st.caption(node.node.get_content()[:500] + "...")
                st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response.response})