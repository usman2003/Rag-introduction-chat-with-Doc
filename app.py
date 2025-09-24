import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import docx
from PyPDF2 import PdfReader

# ---------------- Config ----------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

PERSIST_DIRECTORY = "chroma_storage"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
RETRIEVE_K = 3
SOURCE_SNIPPET_CHARS = 300
MAX_CARRIED_CONTEXT_CHARS = 2000
MAX_CHAT_HISTORY_TURNS = 5  # how many past turns to keep

# ---------------- Embeddings ----------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

# ---------------- LLM ----------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=hf_token,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=hf_token,
    trust_remote_code=True,
    device_map="auto"
)
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# ---------------- Document Loaders ----------------
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def load_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        print(f"Unsupported file type skipped: {file_path}")
        return ""

# ---------------- Helpers ----------------
def split_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def create_vectorstore(all_chunks):
    vectordb = Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    return vectordb

def truncate_text(text, max_chars):
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "…"

# ---------------- Query Logic ----------------
def query_rag_with_history(vectordb, question, chat_history):
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVE_K})
    docs = retriever.get_relevant_documents(question)

    retrieved_texts, sources = [], []
    for d in docs:
        chunk_text = d.page_content
        src = d.metadata.get("source", "unknown")
        retrieved_texts.append(chunk_text)
        sources.append({
            "source": src,
            "snippet": truncate_text(chunk_text.replace("\n", " "), SOURCE_SNIPPET_CHARS)
        })

    retrieved_context = "\n\n".join(retrieved_texts).strip()

    # keep limited history
    short_history = chat_history[-MAX_CHAT_HISTORY_TURNS:]
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in short_history])

    combined_context = (
        f"Conversation history:\n{history_str}\n\n"
        f"Retrieved context:\n{retrieved_context}"
    ).strip()

    combined_context = truncate_text(combined_context, MAX_CARRIED_CONTEXT_CHARS)

    prompt = (
        "You are an assistant answering questions using BOTH the retrieved document context "
        "and the conversation history. "
        "If the answer cannot be found in either, say 'I don't know.' "
        "Keep answers short (1-3 sentences).\n\n"
        f"{combined_context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    raw = llm(prompt, max_length=256, do_sample=False, num_return_sequences=1)
    if isinstance(raw, str):
        answer = raw
    elif isinstance(raw, list):
        first = raw[0]
        if isinstance(first, dict):
            answer = first.get("generated_text") or first.get("text") or str(first)
        else:
            answer = str(first)
    elif isinstance(raw, dict):
        answer = raw.get("generated_text") or raw.get("text") or str(raw)
    else:
        answer = str(raw)

    return answer.strip(), sources

# ---------------- Script Execution ----------------
input_path = input("Enter a file path or folder path: ").strip()

files = []
if os.path.isdir(input_path):
    for fname in os.listdir(input_path):
        if fname.lower().endswith((".txt", ".pdf", ".docx")):
            files.append(os.path.join(input_path, fname))
else:
    files.append(input_path)

all_chunks = []
for file_path in files:
    print(f"Processing {file_path}")
    text = load_document(file_path)
    if text and text.strip():
        all_chunks.extend(split_document(text))

print(f"\nLoaded {len(files)} documents, total {len(all_chunks)} chunks")

vectordb = create_vectorstore(all_chunks)
print("Documents indexed successfully.")

chat_history = []
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Goodbye.")
        break
    if not query:
        print("Please type a question or 'exit'.")
        continue

    answer, sources = query_rag_with_history(vectordb, query, chat_history)
    chat_history.append((query, answer))

    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Sources (short snippets) ===")
    for s in sources:
        print(f"- {s['source']}: {s['snippet']}")
    print("\n(Conversation history carried forward automatically.)")
