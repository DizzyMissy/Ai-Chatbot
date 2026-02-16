from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import faiss
import numpy as np
from pypdf import PdfReader
import uuid

client = OpenAI()
app = FastAPI()

# ===== Memory per session =====
sessions = {}

# ===== Vector store =====
dimension = 1536
index = faiss.IndexFlatL2(dimension)
documents = []

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Use the provided context from PDFs when available.
If the answer is not in the context, say you don't know.
Be concise and accurate.
"""

# ===== Helpers =====
def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]

def embed_text(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(emb.data[0].embedding).astype("float32")

def search_docs(query, k=3):
    if len(documents) == 0:
        return ""

    q_emb = embed_text(query)
    D, I = index.search(np.array([q_emb]), k)
    results = [documents[i] for i in I[0] if i < len(documents)]
    return "\n".join(results)

# ===== Models =====
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

# ===== PDF Upload =====
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    for chunk in chunks:
        emb = embed_text(chunk)
        index.add(np.array([emb]))
        documents.append(chunk)

    return {"status": "PDF processed", "chunks": len(chunks)}

# ===== Chat with streaming =====
@app.post("/chat")
def chat(req: ChatRequest):

    session_id = req.session_id or str(uuid.uuid4())
    memory = get_session(session_id)

    context = search_docs(req.message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"}
    ] + memory[-10:] + [{"role": "user", "content": req.message}]

    def stream():
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            temperature=0.3,
            stream=True
        )

        full_reply = ""

        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            full_reply += token
            yield token

        memory.append({"role": "user", "content": req.message})
        memory.append({"role": "assistant", "content": full_reply})

    return StreamingResponse(stream(), media_type="text/plain")
