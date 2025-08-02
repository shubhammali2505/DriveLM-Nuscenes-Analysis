import logging
from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from .config import get_config
from .dataloader import RAGDataLoader

logger = logging.getLogger(__name__)


class RAGRetriever:

    def __init__(self, vectorstore=None):
        self.config = get_config()
        self.vectorstore = vectorstore
        self.retriever = None

    def initialize(self) -> bool:
        try:
            if not self.vectorstore:
                logger.info("No vectorstore passed, loading from disk...")
                loader = RAGDataLoader()
                if not loader.load_vectorstore():
                    logger.warning("Vectorstore not found, trying to create one...")
                    if not loader.load_data():
                        logger.error("❌ Failed to load data for vectorstore creation")
                        return False
                    if not loader.create_vectorstore():
                        logger.error("❌ Failed to create new vectorstore")
                        return False
                self.vectorstore = loader.get_vectorstore()

            if not self.vectorstore:
                logger.error("❌ Vectorstore is still None after initialization")
                return False

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            logger.info("✅ Retriever initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error initializing retriever: {e}")
            return False
        

    def load_index(self) -> bool:
        """Load FAISS index from disk if available."""
        try:
            from langchain.vectorstores import FAISS
            from .embedder import create_embedder

            vectorstore_path = self.config.vector_db_path
            embedder = create_embedder().model.encode   

            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                embedder,
                allow_dangerous_deserialization=True 
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            logger.info(f"✅ FAISS index loaded from {vectorstore_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading FAISS index: {e}")
            return False


    def build_index(self, documents) -> bool:
        """Build FAISS index from documents (after embedding)."""
        try:
            from langchain.vectorstores import FAISS
            from .embedder import create_embedder

            embedder = create_embedder().model.encode  

            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embedder
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            logger.info("✅ FAISS index built from documents")
            return True
        except Exception as e:
            logger.error(f"❌ Error building FAISS index: {e}")
            return False

    def save_index(self) -> bool:
        """Save FAISS index to disk."""
        try:
            if not self.vectorstore:
                logger.error("❌ No vectorstore to save")
                return False
            self.vectorstore.save_local(self.config.vector_db_path)
            logger.info(f"✅ FAISS index saved to {self.config.vector_db_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving FAISS index: {e}")
            return False


    def retrieve(self, question: str, top_k: int = None, strategy: str = "dense") -> List[Dict[str, Any]]:

            try:
                if not self.retriever:
                    logger.warning("Retriever not initialized, calling initialize()...")
                    if not self.initialize():
                        return []

                k = top_k or self.config.top_k

                docs = self.retriever.get_relevant_documents(question)[:k]

                if strategy == "hybrid":
                    logger.warning("⚠️ Hybrid retrieval not implemented yet. Falling back to dense retrieval.")

                retrieved_items = []
                for i, doc in enumerate(docs):
                    item = {
                        "rank": i + 1,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "question": doc.metadata.get("question", ""),
                        "answer": doc.metadata.get("answer", ""),
                        "image_path": doc.metadata.get("image_path", ""),
                        "question_type": doc.metadata.get("question_type", ""),
                        "semantic_intent": doc.metadata.get("semantic_intent", ""),
                        "entities": doc.metadata.get("entities", []),
                        "scene_token": doc.metadata.get("scene_token", ""),
                        "complexity_score": doc.metadata.get("complexity_score", 0),
                    }
                    retrieved_items.append(item)

                logger.info(f"✅ Retrieved {len(retrieved_items)} documents using strategy={strategy}")
                return retrieved_items

            except Exception as e:
                logger.error(f"❌ Error retrieving documents: {e}")
                return []

    def get_statistics(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"total_items": 0, "embedding_dimension": 0, "similarity_metric": "N/A"}

        try:
            return {
                "total_items": len(self.vectorstore.docstore._dict),
                "embedding_dimension": self.vectorstore.index.d,
                "similarity_metric": "cosine",  # fixed
            }
        except Exception:
            return {"status": "error"}



def create_retriever(vectorstore=None) -> RAGRetriever:
    retriever = RAGRetriever(vectorstore)
    return retriever
