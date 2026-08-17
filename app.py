import os
import tempfile
import uuid

import streamlit as st

from rag import get_embeddings, get_llm, build_vectorstore_from_pdf, ask_rag


st.set_page_config(page_title="AskMyPDF", page_icon="", layout="wide")

# Make the Groq key available wherever it was supplied - Streamlit
# Cloud's Secrets panel, or a local .streamlit/secrets.toml.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .subtitle { color: #888; font-size: 0.95rem; margin-top: -0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🗎 AskMyPDF")
st.markdown(
    '<p class="subtitle">Upload a PDF and ask questions about it — answers are grounded only in that document.</p>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


with st.sidebar:
    st.header("Your document")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Reading and indexing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                embeddings = get_embeddings()
                vectorstore, page_count, chunk_count = build_vectorstore_from_pdf(
                    tmp_path,
                    embeddings,
                    collection_name=f"doc_{st.session_state.session_id}",
                )
            finally:
                os.unlink(tmp_path)

        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.messages = []
        st.success(f"Indexed {page_count} pages ({chunk_count} chunks)")

    if st.session_state.pdf_name:
        st.caption(f"Currently loaded: **{st.session_state.pdf_name}**")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def render_sources(documents):
    if not documents:
        st.caption("No sources available.")
        return
    with st.expander(f"Sources ({len(documents)})"):
        seen = set()
        for doc in documents:
            page = doc.metadata.get("page", "Unknown")
            if page in seen:
                continue
            seen.add(page)
            st.markdown(f"**Page {page}**")


if st.session_state.vectorstore is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("documents", []))

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                llm = get_llm()
                answer, documents = ask_rag(question, st.session_state.vectorstore, llm)
            st.markdown(answer)
            render_sources(documents)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "documents": documents}
        )