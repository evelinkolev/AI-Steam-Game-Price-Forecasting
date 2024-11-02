from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from llm_client import NvidiaLLM
from config import Config


class RetrievalChain:
    """Sets up the Retrieval Augmented Generation (RAG) chain."""

    def __init__(self, documents: List[Document]):
        # Initialize embeddings with specific model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Enhanced prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question:

            Context:
            {context}

            Question:
            {question}

            Provide a detailed analysis based on the historical data:
            """
        )

        # Initialize LLM and chain
        self.llm = NvidiaLLM(
            model_name=Config.MODEL_NAME,
            temperature=Config.COMPLETION_PARAMS["temperature"],
            max_tokens=Config.COMPLETION_PARAMS["max_tokens"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG chain with error handling."""
        try:
            return self.chain.invoke({"query": question})  # Changed from question to query
        except Exception as e:
            raise RuntimeError(f"Error during RAG chain query: {str(e)}")