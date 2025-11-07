import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from llm.llm_services import LANGUAGE, get_llm
from storage.db_models import (
    BasketballPlayerPerGame,
    Competition,
    FootballPlayerPerGame,
    Player,
    Sport,
    Team,
)
from storage.vector_store import VectorStoreManager

load_dotenv()

DB_PATH = Path(os.getenv("DB_DIR", "data/storage/db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

console = Console()
vs_manager = VectorStoreManager()
db = SessionLocal()


# ==================================================
# 🛠️ TOOL 1: Search Knowledge Base (Articles)
# ==================================================
@tool
def search_knowledge_base(query: str) -> str:
    """Search internal sports news articles for insights, commentary, or context."""
    try:
        results = vs_manager.query(query, k=3)
        if not results:
            return "No relevant articles found in the knowledge base."
        parts = []
        for r in results:
            title = r["metadata"].get("title", "Untitled")
            content = r["content"][:400].replace("\n", " ")
            parts.append(f"📄 **{title}**: {content}...")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# ==================================================
# 🛠️ TOOL 2: Query Player/Team Stats from DB
# ==================================================
@tool
def query_database_stats(natural_language_query: str) -> str:
    """
    Use this tool to retrieve specific, quantitative statistics for a player or a team from the database.
    It is best for objective data like points per game, goals, assists, rebounds, ratings, etc.
    Example: 'LeBron James points per game' or 'Messi goals last season'.
    """

    try:
        query_lower = natural_language_query.lower()
        player_name_candidate = next(
            (word for word in query_lower.split() if word.istitle() and len(word) > 2),
            None,
        )
        print(query_lower)
        if player_name_candidate:
            player = (
                db.query(Player)
                .filter(Player.name.like(f"%{player_name_candidate}%"))
                .first()
            )
            if player:
                sport_name = (
                    player.team.sport.name.lower()
                    if player.team and player.team.sport
                    else "unknown"
                )
                if "basketball" in sport_name:
                    stats = (
                        db.query(BasketballPlayerPerGame)
                        .filter(BasketballPlayerPerGame.player_id == player.id)
                        .first()
                    )
                    if stats:
                        return (
                            f"🏀 **{player.name}** (Basketball) - Per Game Stats:\n"
                            f"  - Points: {stats.points or 'N/A'}\n"
                            f"  - Rebounds: {stats.rebounds or 'N/A'}\n"
                            f"  - Assists: {stats.assists or 'N/A'}\n"
                            f"  - Steals: {stats.steals or 'N/A'}"
                        )
                # Add your Football logic here if needed
                return f"Player '{player.name}' was found, but no specific stats are available in the database."

        return "Could not find specific stats for the query. Please specify a full player name. For team stats, try searching the knowledge base for recent match analysis."

    except Exception as e:
        return f"An error occurred while querying the database: {str(e)}"
    finally:
        db.close()


def setup_agent():
    model = get_llm()
    current_date = datetime.now().strftime("%Y-%m-%d")
    language = (LANGUAGE,)
    prompt = f"""
You are 'SportSense', a highly knowledgeable and data-driven sports analyst AI. Your goal is to provide insightful, accurate, and up-to-date answers to sports-related questions.
**Your final answer MUST be in the following language: {language}**
You have access to two powerful tools to help you:
1.  `search_knowledge_base`: Use this to get the latest news, expert analysis, commentary, and context on players, teams, and events.
2.  `query_database_stats`: Use this to get hard, quantitative data and statistics for players (e.g., goals per game, points, assists).
**Your Strategy:**
1.  **Analyze the User's Query:** Understand if the user is asking for objective stats (use `query_database_stats`), subjective analysis/news (use `search_knowledge_base`), or a combination of both.
2.  **Use Tools Strategically:**
    - For questions like "How has Messi been playing for Inter Miami?", you should FIRST `search_knowledge_base` to get recent news and analysis. You might then follow up with `query_database_stats` to fetch his recent stats to support the analysis.
    - For questions like "What are LeBron James's points per game?", you should directly use `query_database_stats`.
3.  **Synthesize, Don't Just Report:** Do not just dump the raw output from the tools. Combine the information into a coherent, well-written answer. For example, if the knowledge base says a player is in "great form," support this claim with their recent stats from the database.
4.  **Be Clear:** If you can't find information, say so. Don't make things up.
Today's Date is: {current_date}
Begin your thought process below to answer the user's question. Use the tools available to you.
"""
    agent = create_agent(
        model,
        system_prompt=prompt,
        tools=[search_knowledge_base, query_database_stats],
        checkpointer=InMemorySaver(),
    )

    return agent


# ==================================================
# 🖥️ Main Chat Loop
# ==================================================
def llm_chat():
    console.print(
        Panel("⚡ Sports Insight Agent: Articles + Stats", border_style="blue")
    )
    agent = setup_agent()

    try:
        while True:
            console.print(Rule(title="You", style="blue"))
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            if not user_input or user_input.lower() in ["quit", "exit", "q"]:
                break

            agent_input = {
                "messages": [{"role": "user", "content": user_input}],
            }
            # Stream output
            displayed = ""
            panel = Panel(
                Markdown(""), title="[green]Assistant[/green]", border_style="green"
            )
            with Live(panel, console=console, refresh_per_second=12) as live:
                for token, metadata in agent.stream(
                    agent_input,
                    config={"thread_id": "1"},
                    stream_mode="messages",
                ):
                    if metadata["langgraph_node"] == "model":
                        displayed += token.content
                        live.update(
                            Panel(
                                Markdown(displayed),
                                title="[green]Assistant[/green]",
                                border_style="green",
                            )
                        )

    except KeyboardInterrupt:
        console.print("\n[red]Exited by user.[/red]")
    console.print("[yellow]Goodbye! May your bets be wise. 🍀[/yellow]")


if __name__ == "__main__":
    llm_chat()
