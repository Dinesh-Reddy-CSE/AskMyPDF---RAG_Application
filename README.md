# 🗎 AskMyPDF

Upload a PDF, ask it questions, get answers grounded only in that document — no hallucinated facts from outside the file, and if the answer isn't in there, it says so instead of guessing.

<p>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/langchain-orchestration-1C3C3C" alt="LangChain">
  <img src="https://img.shields.io/badge/groq-inference-F55036" alt="Groq">
  <img src="https://img.shields.io/badge/chromadb-vector%20store-6C4FFF" alt="ChromaDB">
  <img src="https://img.shields.io/badge/huggingface-embeddings-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

**[Try it live →](https://ragapplication-ugpxyfswbr8pgzjwdikxyy.streamlit.app)**

> Built as a learning project to get hands-on with RAG pipelines, vector search, and deploying an LLM-backed app end to end. Not production-hardened — treat it accordingly.

## What it does

Drop a PDF into the sidebar and the app chunks it, embeds those chunks locally, and stores them in a vector database for that session only. Every question you ask gets matched against the PDF's own content first — the answer is built strictly from what it retrieves, not from the model's general knowledge. Ask something the PDF doesn't cover and you'll get an honest "not in this document" instead of a made-up answer.

Each visitor's upload is isolated from everyone else's — your PDF's content never gets mixed into someone else's session, even though the app itself is shared.

## How it's built

| Piece | What it's doing here |
|---|---|
| **Streamlit** | The whole UI — file upload, chat interface, session handling |
| **PyPDF** (via LangChain) | Pulls text out of the uploaded PDF, page by page |
| **LangChain text splitters** | Breaks pages into overlapping ~1000-character chunks so retrieval doesn't miss context that spans paragraph boundaries |
| **Sentence-Transformers (all-MiniLM-L6-v2)** | Turns chunks into embeddings, running locally — no API call for this part |
| **ChromaDB** | In-memory vector store, one collection per session, cosine similarity |
| **Groq (openai/gpt-oss-20b)** | Generates the final answer from the retrieved chunks — fast enough that the chat doesn't feel like it's waiting on anything |

## Running it locally

```bash
git clone https://github.com/Dinesh-Reddy-CSE/RAG_Application.git
cd RAG_Application

python -m venv venv
venv\Scripts\activate      # or source venv/bin/activate on macOS/Linux

pip install -r requirements.txt
```

Grab a free API key from [console.groq.com](https://console.groq.com), then create `.streamlit/secrets.toml` (copy the `.example` file that's already in the repo and drop your key in):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Then just:

```bash
streamlit run app.py
```

It'll open at `localhost:8501`. Upload a PDF and start asking.

## Deploying

This runs on [Streamlit Community Cloud](https://share.streamlit.io) for free — point it at this repo, set `app.py` as the entry point, and add `GROQ_API_KEY` under the app's Secrets settings. No server to manage, no Dockerfile needed.

## A couple of things worth knowing

- The vector store lives in memory, not on disk, so it resets whenever the app restarts (Streamlit Cloud free tier sleeps after inactivity). That's intentional — each visitor brings their own PDF, so there's nothing worth persisting between sessions anyway.
- First load can be a little slow since the embedding model downloads on cold start. After that it's cached for the life of the app instance.
- Answers lean conservative on purpose. If a question is only loosely related to the PDF's content, it'll often decline rather than stretch the context to answer it — that trade-off is deliberate, not a bug.

## Project structure

```
RAG_Application/
├── app.py                          # Streamlit UI
├── rag.py                          # indexing + retrieval + prompt logic
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
└── .gitignore
```

## Questions or issues

This was built solo as an educational project, so there may be rough edges. If you run into a bug, have a question about how something works, or just want to talk about it, feel free to open an issue here or reach out directly.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anumula%20Dinesh%20Reddy-0A66C2?logo=linkedin&logoColor=white)](https://in.linkedin.com/in/anumula-dinesh-reddy)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for the full text. In short: use it, modify it, learn from it, just keep the copyright notice attached.