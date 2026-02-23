import os
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

# NEW: PDF support
from pypdf import PdfReader

# -------------------- ENV + LOGGING SETUP --------------------

load_dotenv()

GOOGLE_KEY = os.getenv("API_KEY")
if not GOOGLE_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in environment.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("rag_agent.log"),   # Persist logs
        logging.StreamHandler()                 # Console logs
    ]
)
logger = logging.getLogger("RAG-Agent")

# -------------------- RAG SERVICE --------------------

class RAGService:

    def __init__(self):
        logger.info("Initializing RAG Service...")

        # LLM client
        self.client = genai.Client(api_key=GOOGLE_KEY)

        # Embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_KEY
        )

        # Vector DB (persistent on disk)
        self.qdrant = QdrantClient(path="./qdrant_storage")

        # Create collections if missing
        self._init_collections()

        # Main knowledge store (Tier-2 RAG)
        self.doc_store = QdrantVectorStore(
            client=self.qdrant,
            collection_name="enterprise_docs",
            embedding=self.embeddings
        )

        # Semantic cache (Tier-1)
        self.cache_store = QdrantVectorStore(
            client=self.qdrant,
            collection_name="semantic_cache",
            embedding=self.embeddings
        )

        self.llm_model = os.getenv("LLM_MODEL")
        logger.info("RAG Service initialized successfully.")
        logger.info(f"Using LLM Model: {self.llm_model}")

    # -------------------- VECTOR COLLECTION INIT --------------------

    def _init_collections(self):
        try:
            collections = [c.name for c in self.qdrant.get_collections().collections]

            if "enterprise_docs" not in collections:
                logger.info("Creating collection: enterprise_docs")
                self.qdrant.create_collection(
                    "enterprise_docs",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

            if "semantic_cache" not in collections:
                logger.info("Creating collection: semantic_cache")
                self.qdrant.create_collection(
                    "semantic_cache",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

        except Exception as e:
            logger.exception("Failed to initialize Qdrant collections.")
            raise RuntimeError("Vector DB initialization failed") from e

    # -------------------- INGESTION (TXT + PDF) --------------------
    # CHANGE: Added PDF parsing, chunking, logging, and error handling

    def ingest(self, directory: str):
        logger.info(f"Starting ingestion from directory: {directory}")

        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(directory)

        documents = []

        for file in os.listdir(directory):
            path = os.path.join(directory, file)

            try:
                text = ""

                # ---- TXT FILE SUPPORT ----
                if file.lower().endswith(".txt"):
                    logger.info(f"Reading TXT file: {file}")
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()

                # ---- PDF FILE SUPPORT (NEW) ----
                elif file.lower().endswith(".pdf"):
                    logger.info(f"Reading PDF file: {file}")
                    reader = PdfReader(path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                else:
                    logger.info(f"Skipping unsupported file: {file}")
                    continue

                if not text.strip():
                    logger.warning(f"No extractable text in file: {file}")
                    continue

                # ---- CHUNKING ----
                chunks = []
                for i in range(0, len(text), 1000):
                    chunk = text[i:i+1000]
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={"source": file}
                    ))

                logger.info(f"Created {len(chunks)} chunks from {file}")
                documents.extend(chunks)

            except Exception as e:
                logger.exception(f"Failed to process file: {file}")
                continue   # Do not stop ingestion if one file fails

        if not documents:
            logger.warning("No documents were ingested.")
            return

        try:
            self.doc_store.add_documents(documents)
            logger.info(f"Successfully ingested {len(documents)} chunks into vector store.")
        except Exception as e:
            logger.exception("Failed to store documents in Qdrant.")
            raise RuntimeError("Vector store ingestion failed") from e

    # -------------------- TIER 1: SEMANTIC CACHE --------------------

    def check_cache(self, query: str) -> Optional[str]:
        """Checks if this question was asked before."""
        try:
            logger.info(f"Checking Semantic Cache for: '{query}'")
            
            # 1. Search with score
            results = self.cache_store.similarity_search_with_score(query, k=1)
            
            if results:
                doc, score = results[0]
                logger.info(f"Found candidate. Score: {score}")  # <--- NEW DEBUG LOG
                
                # 2. Lower threshold slightly to 0.75 (75% match)
                if score > 0.75: 
                    logger.info(f"Tier 1: Semantic cache Hit! (Score {score:.4f})")
                    return doc.metadata.get("answer")
                else:
                    logger.info("Candidate score too low. Ignoring.")
            else:
                logger.info("No candidates found in cache.")
                
        except Exception as e:
            logger.warning(f"Semantic Cache check failed: {e}")
        return None

    def save_cache(self, query: str, answer: str):
        try:
            doc = Document(page_content=query, metadata={"answer": answer})
            self.cache_store.add_documents([doc])
            logger.info("Saved answer to semantic cache.")
        except Exception:
            logger.exception("Failed to save to semantic cache.")

    # -------------------- TIER 2: VECTOR RETRIEVAL --------------------

    def retrieve_answer(self, query: str, k: int = 5):
        try:
            results = self.doc_store.similarity_search(query, k=k)
            logger.info(f"Vector retrieval returned {len(results)} chunks.")

            contexts = []
            sources = set()

            for doc in results:
                contexts.append(doc.page_content)
                sources.add(doc.metadata.get("source", "unknown"))

            return contexts, list(sources)

        except Exception:
            logger.exception("Vector retrieval failed.")
            return [], []

    # -------------------- GENERATION --------------------

    def generate_answer(self, query: str):
        start = time.time()
        logger.info(f"Received query: {query}")

        # ---- Tier 1: Semantic Cache ----
        cached = self.check_cache(query)
        if cached:
            return cached, [], "Semantic Cache", time.time() - start

        # ---- Tier 2: Vector RAG ----
        contexts, sources = self.retrieve_answer(query)

        if not contexts:
            logger.warning("No relevant context found in vector DB.")
            return "No relevant information found in knowledge base.", [], "No Retrieval", time.time() - start

        context_block = "\n\n".join(contexts)

#         prompt = f"""
# You are an enterprise assistant.
# Answer strictly from the context below.
# If the answer is not present, say: "Answer not present in knowledge base."

# Context:
# {context_block}

# Question: {query}
# """

        prompt = f"""
You are an expert enterprise assistant.

Use the following retrieved context as factual evidence.
You must:
1. Understand the information.
2. Reason over it.
3. Combine relevant parts.
4. Produce a complete, well-formed answer in your own words.
5. Do not hallucinate beyond the context.

Context:
{context_block}

Question: {query}

Answer (well-structured, synthesized, grounded in the context):
"""
        try:
            logger.info("Calling Gemini LLM...")
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            final_answer = response.text
            logger.info("LLM response generated successfully.")

            # Save to semantic cache
            self.save_cache(query, final_answer)

            return final_answer, sources, "Vector RAG", time.time() - start

        except Exception:
            logger.exception("LLM generation failed.")
            return "LLM service error. Please try again later.", [], "LLM Error", time.time() - start
