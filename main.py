from pathlib import Path
import pandas as pd
import streamlit as st
from typing import List
from data_loader import DataLoader
from rate_limiter import RateLimiter
from retrieval_chain import RetrievalChain
from steam_scraper import SteamScraper
from datetime import datetime


class GameInsightApp:
    """Sets up the Streamlit App."""


    def __init__(self):

        self.rate_limiter = RateLimiter(
            redis_url=st.secrets["UPSTASH_REDIS_URL"],
            redis_token=st.secrets["UPSTASH_REDIS_TOKEN"]
        )


        st.set_page_config(
            page_title="GameInsight Chat",
            page_icon="🎮",
            layout="wide",
            initial_sidebar_state="collapsed")

        st.markdown("""
                    <style>
                    .main {
                        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%);
                        color: #e0e0e0;
                        font-family: 'Segoe UI', sans-serif;
                    }
                    .stButton>button {
                        background-color: #4a4e69;
                        color: white;
                        border-radius: 8px;
                        transition: all 0.3s ease;
                    }
                    .stButton>button:hover {
                        background-color: #6b7280;
                        transform: scale(1.05);
                    }
                    .stChatInput {
                        border-radius: 10px;
                        background-color: rgba(255, 255, 255, 0.05);
                        color: white;
                    }
                    .chat-container {
                        background-color: rgba(255, 255, 255, 0.05);
                        border-radius: 10px;
                        padding: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    h1, h2 {
                        color: #c7d5e0;
                    }
                    </style>
                """, unsafe_allow_html=True)


        self.scraper = SteamScraper()
        self.ensure_data()

        # Initialise session state variables
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = self.load_rag_chain()
        if 'suggested_questions' not in st.session_state:
            st.session_state.suggested_questions = []

    def analyse_data(self) -> List[str]:
        """Analyse the CSV data and generate relevant questions."""
        df = pd.read_csv(self.scraper.output_path)
        questions = []

        # Analyse player counts
        max_players = df['current_players'].max()
        if max_players > 100000:
            questions.append(f"What game has the highest current player count and why might it be so popular?")

        # Analyse price distributions
        free_games = df[df['price'] == 0].shape[0]
        if free_games > 0:
            questions.append(f"What are the most popular free-to-play games right now?")

        # Analyse player trends
        peak_vs_current = df[df['peak_players_today'] > df['current_players'] * 2]
        if not peak_vs_current.empty:
            questions.append("Which games show the biggest difference between peak and current players today?")

        # Add general analytical questions
        questions.extend([
            "What are the current trending games based on player count growth?",
            "Which price range shows the highest player engagement?",
            "What patterns do you notice in player activity across different game genres?",
        ])

        return questions

    def ensure_data(self):
        """Ensure valid game data is available."""

        def is_valid_file(path):
            if not path.is_file() or path.stat().st_size == 0:
                return False

            required_columns = ['name', 'price', 'current_players', 'peak_players_today', 'date']
            try:
                df = pd.read_csv(path, usecols=required_columns, nrows=5)
                return not df.isnull().any().any()
            except (ValueError, pd.errors.ParserError, FileNotFoundError):
                return False

        try:
            self.scraper.scrape()
            if is_valid_file(Path('games_fresh.csv')):
                self.scraper.output_path = 'games_fresh.csv'
                return
        except Exception:
            pass

        if is_valid_file(Path('games_back-up.csv')):
            self.scraper.output_path = 'games_back-up.csv'
            return

        raise RuntimeError("Unable to retrieve valid data: scraping and backup both failed.")

    def load_rag_chain(self) -> RetrievalChain:
        data_path = self.scraper.output_path
        data = DataLoader.load_data(data_path)
        documents = DataLoader.create_documents(data)
        return RetrievalChain(documents)

    def run(self):

        st.markdown("<h1 style='text-align: center;'>🎮 GameInsight Chat</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a3bffa;'>Explore gaming trends and stats in real-time!</p>", unsafe_allow_html=True)

        col1, col2 = st.columns([0.55, 0.45], gap="medium")

        with col1:

            # Generate and display suggested questions
            if not st.session_state.suggested_questions:
                st.session_state.suggested_questions = self.analyse_data()

            with st.container():
                st.subheader("🔍 Suggested Questions")
                for question in st.session_state.suggested_questions:
                    if st.button(question, key=f"btn_{hash(question)}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        with st.spinner("Processing your query..."):
                            try:
                                result = st.session_state.rag_chain.query(question)
                                response = result["result"]
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.session_state.current_sources = result["source_documents"]
                            except Exception as e:
                                error_message = f"An error occurred: {e}"
                                st.session_state.messages.append({"role": "assistant", "content": error_message})

            # Handle user input
            if prompt := st.chat_input("Ask about game statistics..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("Processing your query..."):
                    try:
                        result = st.session_state.rag_chain.query(prompt)
                        response = result["result"]
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.current_sources = result["source_documents"]
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

        with col2:
            st.markdown("### 📜 Chat History")
            with st.container(height=500, border=True):
                for message in st.session_state.messages:
                    avatar = "🤖" if message["role"] == "assistant" else "👤"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.write(f"**{message['role'].capitalize()}** | {datetime.now().strftime('%H:%M:%S')}")
                        st.write(message["content"])


if __name__ == "__main__":
    app = GameInsightApp()
    app.run()
