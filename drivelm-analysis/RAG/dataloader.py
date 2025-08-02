# RAG/dataloader.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from .config import get_config
from .embedder import create_embedder

logger = logging.getLogger(__name__)


class RAGDataLoader:

    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)

        self.documents: List[Document] = []
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name= self.config.embedding.model_name,
            model_kwargs={'device':  self.config.embedding.device}
        )

    def load_data(self) -> bool:
        try:
            data_path = Path(self.config.unified_data_path)
            if not data_path.exists():
                logger.error(f"❌ Data file not found: {data_path}")
                return False

            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            items = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
            logger.info(f"✅ Loaded {len(items)} items from {data_path}")

            self.documents = []
            for i, item in enumerate(items):
                doc = self._create_document(item, i)
                if doc:
                    self.documents.append(doc)

            logger.info(f"✅ Created {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

    def _create_document(self, item: Dict[str, Any], index: int) -> Document:
        q_data = item.get("question", {})
        question = q_data.get("question", "")
        answer = q_data.get("answer", "")

        content = f"Q: {question}\nA: {answer}"

        metadata = {
            "id": item.get("id", f"item_{index}"),
            "question": question,
            "answer": answer,
            "question_type": q_data.get("question_type", "unknown"),
            "image_path": q_data.get("image_path", ""),
            "scene_token": q_data.get("scene_token", ""),
            "semantic_intent": q_data.get("semantic_intent", ""),
            "entities": q_data.get("extracted_entities", []),
            "keywords": q_data.get("keywords", []),
            "complexity_score": q_data.get("complexity_score", 0),
        }

        return Document(page_content=content, metadata=metadata)

    def create_vectorstore(self) -> bool:
        """Build FAISS vectorstore."""
        try:
            if not self.documents:
                logger.error("❌ No documents to create vectorstore")
                return False

            logger.info("🔄 Creating FAISS vectorstore...")
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings 
            )

            vdb_path = Path(self.config.vector_db_path)
            vdb_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(vdb_path))
            logger.info(f"✅ Vectorstore created and saved to {vdb_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            return False

    def load_vectorstore(self) -> bool:
        """Load existing FAISS vectorstore."""
        try:
            vdb_path = Path(self.config.vector_db_path)
            if not vdb_path.exists():
                logger.error(f"❌ Vectorstore not found: {vdb_path}")
                return False

            self.vectorstore = FAISS.load_local(
                str(vdb_path),
                embeddings=self.embedder,
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ Vectorstore loaded from {vdb_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading vectorstore: {e}")
            return False
        
    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {"error": "No documents available"}

        stats = {
            "total_items": len(self.documents),
            "items_with_images": sum(1 for d in self.documents if d.metadata.get("image_path")),
            "avg_text_length": sum(len(d.page_content) for d in self.documents) / len(self.documents),
        }

        q_types = [d.metadata.get("question_type", "unknown") for d in self.documents]
        stats["question_types"] = {qt: q_types.count(qt) for qt in set(q_types)}

        return stats

    def get_vectorstore(self):
        return self.vectorstore


def create_dataloader(config_override: Dict[str, Any] = None) -> RAGDataLoader:
    return RAGDataLoader(config_override)
