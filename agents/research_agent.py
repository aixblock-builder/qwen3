# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from utils.qwen_llm import QwenLLM
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with the OpenAI model."""
        self.llm = QwenLLM()
        self.prompt_template = (
            "Answer the following question based on the provided context. Be precise and factual.\n"
            "Question: {question}\n"
            "Context:\n{context}\n"
            "If the context is insufficient, respond with: 'I cannot answer this question based on the provided documents.'"
        )
        
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = self.prompt_template.format(question=question, context=context)
        try:
            answer = self.llm.generate(prompt)
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
        
        return {
            "draft_answer": answer,
            "context_used": context
        }