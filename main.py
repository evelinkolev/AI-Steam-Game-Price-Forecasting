import logging
from data_loader import DataLoader
from retrieval_chain import RetrievalChain
from pathlib import Path


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to execute the RAG workflow."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate data file path
        data_path = Path('2024-11-02_Steam_top100_most_played_games_clean.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # Load and prepare data
        logger.info("Loading data...")
        data = DataLoader.load_data(str(data_path))
        documents = DataLoader.create_documents(data)

        # Set up the RAG chain
        logger.info("Initializing RAG chain...")
        rag_chain = RetrievalChain(documents)

        # Example query
        question = "Are games with similar player counts priced similarly, and how do they compete on other factors?"
        logger.info(f"Executing query: {question}")

        result = rag_chain.query(question)

        # Display results
        print("\nGenerated Response:")
        print("-" * 80)
        print(result["result"])
        print("\nRetrieved Documents for Context:")
        print("-" * 80)
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()