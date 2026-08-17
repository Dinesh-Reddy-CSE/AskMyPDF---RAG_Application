import re

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# A model name on the Hugging Face Hub (not a local path) so this
# downloads automatically wherever the app runs, including on
# Streamlit Cloud where a local D:\ path wouldn't exist.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# openai/gpt-oss-20b is Groq's current recommended replacement for
# the now-deprecated llama-3.1-8b-instant: fast, and on the free tier.
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6

# Cosine distance (0 = identical, 2 = opposite). See the indexing
# function below - collection_metadata forces cosine so this scale
# stays predictable regardless of what PDF gets uploaded.
SIMILARITY_THRESHOLD = 0.8

FALLBACK_MESSAGE = (
    "Sorry, I'm not able to answer that using the uploaded document."
)


# ---------------------------------------------------------------
# Cached resources
#
# These are loaded once per app process and reused across every
# visitor's session - they hold no per-user data, just the model
# weights / API client, so sharing them is safe and avoids reloading
# a ~90MB embedding model on every rerun.
# ---------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------
# Indexing an uploaded PDF
#
# No persist_directory is passed to Chroma, so each vectorstore
# lives only in memory for this process. Each session gets its own
# collection_name (see app.py), so one visitor's PDF can never leak
# into another visitor's answers even though the app process itself
# is shared.
# ---------------------------------------------------------------

def build_vectorstore_from_pdf(pdf_path: str, embeddings, collection_name: str):
    documents = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectorstore, len(documents), len(chunks)


# ---------------------------------------------------------------
# Code policy
#
# Patterns are anchored to the whole line and require 2+ matches
# before flagging an answer as code, so ordinary prose (e.g. "For
# beginners in this topic...") doesn't get misclassified.
# ---------------------------------------------------------------

CODE_LINE_PATTERNS = [
    r"^\s*def\s+\w+\s*\([^)]*\)\s*:\s*$",
    r"^\s*class\s+\w+\s*(\([^)]*\))?\s*:\s*$",
    r"^\s*import\s+[\w\.]+\s*$",
    r"^\s*from\s+[\w\.]+\s+import\s+[\w\*,\s]+$",
    r"^\s*for\s+\w+\s+in\s+[\w\.\(\)\[\]:,\s]+:\s*$",
    r"^\s*while\s+[^:]+:\s*$",
    r"^\s*if\s+[^:]+:\s*$",
    r"^\s*elif\s+[^:]+:\s*$",
    r"^\s*else\s*:\s*$",
    r"^\s*try\s*:\s*$",
    r"^\s*except\b[^:]*:\s*$",
    r"^\s*finally\s*:\s*$",
    r"^\s*return\s+.+$",
    r"^\s*print\s*\(.*\)\s*$",
    r"^\s{4,}\S",  # indented line, characteristic of a code block
]


def user_requested_code(question: str) -> bool:
    return "code" in question.lower().split()


def contains_code(text: str) -> bool:
    if "```" in text:
        return True

    matches = sum(
        1
        for pattern in CODE_LINE_PATTERNS
        for _ in re.finditer(pattern, text, flags=re.MULTILINE)
    )
    return matches >= 2


def remove_code(answer: str) -> str:
    answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    answer = re.sub(r"`[^`]*`", "", answer)
    return answer.strip()


def clean_answer(answer: str, question: str) -> str:
    if answer is None:
        return FALLBACK_MESSAGE

    answer = str(answer).strip()
    if not answer:
        return FALLBACK_MESSAGE

    if user_requested_code(question):
        answer = re.sub(r"```\w*\s*", "", answer)
        answer = answer.strip()
        return answer if answer else FALLBACK_MESSAGE

    answer = remove_code(answer)
    if not answer:
        return FALLBACK_MESSAGE
    if contains_code(answer):
        return FALLBACK_MESSAGE

    return answer


# ---------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------

def build_prompt(context: str, question: str) -> str:
    if user_requested_code(question):
        code_instruction = """
CODE POLICY:
The user's question contains the word "code".
1. Return ONLY the requested code.
2. Do NOT explain the code, and no introduction or conclusion.
3. Do NOT wrap the code in Markdown fences.
4. Use ONLY information contained in the context.
5. Do NOT invent code using information not present in the context.
6. If the requested code cannot be answered using the context,
   return the fallback message exactly.
"""
    else:
        code_instruction = """
NO-CODE POLICY:
The user's question does NOT contain the word "code".
1. Do NOT generate code, syntax examples, or Markdown code blocks.
2. Explain using plain text only.
3. Use ONLY information contained in the context.
4. Do NOT use outside knowledge, guess, or infer missing information.
5. If the answer is not clearly supported by the context,
   return the fallback message exactly.
"""

    return f"""
You are a strict Retrieval-Augmented Generation assistant.
You answer questions about the document the user uploaded.
Your knowledge is LIMITED to the CONTEXT provided below.

STRICT RULES:
1. Use ONLY the provided context.
2. Never use outside knowledge or guess.
3. Never invent or assume information not explicitly supported by the context.
4. If the answer is not clearly present in the context, respond EXACTLY with:
{FALLBACK_MESSAGE}
5. Do not add examples unless they are explicitly available in the context.
6. Keep the answer concise and directly related to the question.
7. Follow the code policy below exactly.

{code_instruction}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# ---------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------

def ask_rag(question: str, vectorstore, llm):
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    if not results:
        return FALLBACK_MESSAGE, []

    relevant_results = [
        (document, score) for document, score in results if score <= SIMILARITY_THRESHOLD
    ]

    if not relevant_results:
        return FALLBACK_MESSAGE, []

    documents = [document for document, _ in relevant_results]

    context = "\n".join(
        f"DOCUMENT {i + 1}\nPAGE: {document.metadata.get('page', 'Unknown')}\n\n{document.page_content}"
        for i, document in enumerate(documents)
    )

    prompt = build_prompt(context=context, question=question)

    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        print(f"LLM error: {e}")
        return FALLBACK_MESSAGE, documents

    answer = clean_answer(answer=answer, question=question)

    return answer, documents