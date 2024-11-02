from typing import List
import pandas as pd
from langchain_core.documents import Document


class DataLoader:
    """Class for loading and processing CSV data into LangChain Documents."""

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load and preprocess game data from a CSV file."""
        try:
            data = pd.read_csv(filepath)
            data['current_players'] = pd.to_numeric(data['current_players'], errors='coerce')
            data['peak_players_today'] = pd.to_numeric(data['peak_players_today'], errors='coerce')
            data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    @staticmethod
    def create_documents(data: pd.DataFrame) -> List[Document]:
        """Convert data rows into LangChain Document format."""
        documents = []
        for _, row in data.iterrows():
            try:
                content = (
                    f"Game: {row['name']}, Price: {row['price']}, "
                    f"Current Players: {row['current_players']}, "
                    f"Peak Players Today: {row['peak_players_today']}, "
                    f"Date: {row['date'].strftime('%Y-%m-%d')}"
                )
                doc = Document(
                    page_content=content,
                    metadata={
                        "name": row['name'],
                        "date": row['date'].isoformat(),
                        "price": row['price'],
                        "current_players": row['current_players'],
                        "peak_players_today": row['peak_players_today']
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Skipping row due to error: {str(e)}")
        return documents