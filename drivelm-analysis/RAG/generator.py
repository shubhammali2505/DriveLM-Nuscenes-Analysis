"""
RAG Generator - Gemini Version
"""

import logging
from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from .config import get_config
import os

logger = logging.getLogger(__name__)

class RAGGenerator:
    """Wrapper around Google Gemini LLM for answer generation."""

    def __init__(self):
        self.config = get_config()
        # api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDzRM_Hqc0DIV-eg3MrOo_eUGc6MdceqDo")
        genai.configure(api_key=self.config.llm.api_key)

        # Use model name from config
        self.model = genai.GenerativeModel(self.config.llm.model_name)

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a driving assistant.
Use the retrieved context to answer the user question.
Keep the answer short and natural (2 to 3 sentence).

Context:
{context}

Question: {question}
Answer:""",
        )

    def initialize_text_model(self) -> bool:
        try:
            logger.info(f"✅ Connected to Gemini model.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {e}")
            return False

    def initialize_vision_model(self) -> bool:
        logger.warning("⚠️ Vision model not supported yet in Gemini.")
        return False

    def generate_answer(
        self,
        question: str,
        retrieved_items: List[Dict[str, Any]],
        use_vision: bool = False,
    ) -> Dict[str, Any]:
        """Generate final answer using Gemini API"""
        try:
            context = "\n".join([item.get("content", "") for item in retrieved_items])
            prompt = self.prompt.format(context=context, question=question)

            response = self.model.generate_content(prompt)
            answer_text = response.text.strip() if response else "No answer generated."

            return {
                "answer": answer_text,
                "confidence": 0.9,
                "sources_used": len(retrieved_items),
                "retrieved_contexts": retrieved_items,
                "images": [
                    {"path": item.get("image_path"), "rank": i + 1}
                    for i, item in enumerate(retrieved_items)
                    if item.get("image_path")
                ],
            }

        except Exception as e:
            logger.error(f"❌ Error generating answer with Gemini: {e}")
            return {
                "answer": "Failed to generate answer.",
                "confidence": 0.0,
                "sources_used": 0,
                "retrieved_contexts": [],
                "images": [],
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Return info about the Gemini model being used"""
        return {
            "text_model_loaded": True,
            "vision_model_loaded": False,
            "device": "google-cloud",
            "gemini_model": "gemini-1.5-flash",
        }


def create_generator() -> RAGGenerator:
    return RAGGenerator()
