from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from llm_client import NvidiaLLM
from config import Config


class GamingBuddyPrompt:

    TEMPLATE = """
    Hey mate! *awkward but endearing virtual fist bump* 🎮 You know how we get carried away chatting absolute nonsense between games? Well, brace yourself, because I’ve been snooping around in my gaming notes and found some juicy bits!

    Here’s the good stuff I dug up:
    {context}

    And you're asking:
    {question}

    *leans in dramatically*

    So here’s the deal – I’m going to explain this as if we were lounging on a couch, controllers in hand, snacks precariously balanced on our laps. Expect a few wild tangents, maybe a completely unnecessary analogy, and definitely some obscure gaming references. Sound good? Sweet as!

    Right, let's crack into it:
    """

    @classmethod
    def get_template_with_examples(cls):
        """Returns the template with example responses style guide."""
        return {
            "template": cls.TEMPLATE,
            "response_style_guide": """
            Response Style Examples:
            - Use playful, offbeat humour with a touch of chaos
            - Throw in some unexpected but charming metaphors
            - Keep it breezy: "Right, here’s the deal…"
            - Sprinkle in dramatic storytelling: "Picture this… you’re in the middle of an epic battle..."
            - Make use of quirky enthusiasm: "Mate, this is bonkers!"
            - Keep it casual but sneakily insightful
            """,
            "example_response": """
            Oi, listen to this – this is like discovering a secret boss battle when you thought the game was over!

            Looking at the data, I can tell you that [specific insight]. It’s like uncovering that last piece of the puzzle in a detective RPG – totally satisfying, right?

            But wait – *plot twist!* The numbers ALSO reveal [unexpected finding]. Absolute madness. It’s like thinking you’ve mastered a game, and then BAM! - new DLC drops and turns everything on its head! 😆

            Wild stuff, my friend. Want me to dig deeper? I can ramble on about this all day – like an NPC stuck in a dialogue loop!
            """
        }


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
            search_type="mmr",  # Changed from similarity to MMR
            search_kwargs={
                "k": 8,  # Increased from 5 to 8
                "fetch_k": 20,  # Fetch more candidates for MMR to choose from
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        # Enhanced prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=GamingBuddyPrompt.TEMPLATE
        )

        # Optional: Add style guide to the chain's metadata
        self.style_guide = GamingBuddyPrompt.get_template_with_examples()

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