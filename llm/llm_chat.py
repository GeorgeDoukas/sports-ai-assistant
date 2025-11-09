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

DB_DIR = Path(os.getenv("DB_DIR", "data/storage/db"))
DB_PATH = DB_DIR / "stats.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

console = Console()
vs_manager = VectorStoreManager()
db = SessionLocal()


# ==================================================
# TOOL 1: Search Knowledge Base (Articles)
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
# TOOL 2: Query Player/Team Stats from DB
# ==================================================
@tool
def query_database_stats(player_name: str, metric: str = "all") -> str:
    """
    Use this tool to retrieve specific, quantitative statistics for a player from the database.
    The LLM must extract the exact 'player_name' (e.g., 'Harden J.', 'Messi L.') 
    and the 'metric' (e.g., 'Points', 'Goals', 'all' for all available stats) from the user's natural language query 
    and pass them as arguments to this function.
    """
    try:
        with SessionLocal() as db_session:
            # try:
            # 1. Search for Player using the name extracted by the LLM
            player = (
                db_session.query(Player)
                .filter(Player.name.like(f"%{player_name}%"))
                .first()
            )

            if not player:
                return f"Could not find a player matching '{player_name}' in the database."

            # 2. Determine Sport and Fetch Stats
            sport_name = (
                player.team.sport.name.lower()
                if player.team and player.team.sport
                else "unknown"
            )
            
            output_lines = []

            # --- BASKETBALL LOGIC ---
            if "basketball" in sport_name:
                stats_model = BasketballPlayerPerGame
                stats = (
                    db_session.query(stats_model)
                    .filter(stats_model.player_id == player.id)
                    .first()
                )
                
                if stats:
                    output_lines.append(f"🏀 **{player.name}** (Basketball) - Per Game Stats:")
                    
                    # Define metric mapping for basketball
                    metric_map = {
                        "points": stats.points, "ποντοι": stats.points,
                        "rebounds": stats.rebounds, "ριμπαουντ": stats.rebounds,
                        "assists": stats.assists, "ασιστς": stats.assists,
                        "steals": stats.steals, "κλεψιματα": stats.steals,
                        "minutes": stats.minutes, "λεπτα": stats.minutes,
                    }

                    if metric.lower() == "all":
                        output_lines.append(f"  - Πόντοι (Points): {stats.points or 'N/A'}")
                        output_lines.append(f"  - Σύνολο ριμπάουντ (Rebounds): {stats.rebounds or 'N/A'}")
                        output_lines.append(f"  - Ασίστς (Assists): {stats.assists or 'N/A'}")
                        output_lines.append(f"  - Λεπτά (Minutes): {stats.minutes or 'N/A'}")
                    elif metric.lower() in metric_map:
                        value = metric_map[metric.lower()]
                        output_lines.append(f"  - {metric.capitalize()}: {value or 'N/A'}")
                    else:
                        output_lines.append(f"Could not find the specific basketball metric '{metric}'.")

            # --- FOOTBALL LOGIC ---
            elif "football" in sport_name:
                # Assuming FootballPlayerPerGame is imported and defined with columns like rating, shots, xg, etc.
                stats_model = FootballPlayerPerGame
                stats = (
                    db_session.query(stats_model)
                    .filter(stats_model.player_id == player.id)
                    .first()
                )

                if stats:
                    output_lines.append(f"⚽ **{player.name}** (Football) - Per Game Stats:")
                    
                    # Define metric mapping for football
                    metric_map = {
                        "rating": stats.rating, "βαθμολογια": stats.rating,
                        "shots": stats.shots, "σουτ": stats.shots,
                        "xg": stats.xg, 
                        "touches": stats.touches, "επαφες": stats.touches,
                        "duels": stats.duels, "μονομαχιες": stats.duels,
                    }
                    
                    if metric.lower() == "all":
                        output_lines.append(f"  - Βαθμολογία (Rating): {stats.rating or 'N/A'}")
                        output_lines.append(f"  - Σουτ (Shots): {stats.shots or 'N/A'}")
                        output_lines.append(f"  - Αναμενόμενα γκολ (xG): {stats.xg or 'N/A'}")
                        output_lines.append(f"  - Επαφές (Touches): {stats.touches or 'N/A'}")
                    elif metric.lower() in metric_map:
                        value = metric_map[metric.lower()]
                        output_lines.append(f"  - {metric.capitalize()}: {value or 'N/A'}")
                    else:
                        output_lines.append(f"Could not find the specific football metric '{metric}'.")
            
            # --- FINAL OUTPUT ---
            if output_lines:
                # If we found stats for either sport
                return "\n".join(output_lines)
            elif player:
                # Player found, but no per-game stats available for their sport
                return f"Player '{player.name}' was found, but no specific per-game stats are available in the database for the sport '{sport_name}'. Try searching the knowledge base for news."

            # Should be caught by the first 'if not player' but kept for safety
            return "Could not find specific stats for the query."

    except Exception as e:
        # Handle any unexpected database or code errors
        return f"An error occurred while querying the database for stats: {type(e).__name__}"

def setup_agent():
    model = get_llm()
    current_date = datetime.now().strftime("%Y-%m-%d")
    language = (LANGUAGE,)
    prompt = f"""
You are 'SportSense', a highly knowledgeable and data-driven sports analyst AI. Your goal is to provide insightful, accurate, and up-to-date answers to sports-related questions.
**Your final answer MUST be in the following language: {language}**
You have access to two powerful tools to help you:
1.  `search_knowledge_base`: Use this to get the latest news, expert analysis, commentary, and context on players, teams, and events.
2.  `query_database_stats`: Use this to get hard, quantitative data and statistics for **a specific player**. **When calling this tool, you MUST provide the full, correct player name as the 'player_name' argument** (e.g., 'Gilgeous-Alexander S.', 'Harden J.') and the name of the statistic as the 'metric' argument (e.g., 'Points', 'Rebounds', 'Assists'). If the user asks for general stats, use 'all' for the 'metric'.
**Your Strategy:**
1.  **Analyze the User's Query:** Understand if the user is asking for objective stats (use `query_database_stats`), subjective analysis/news (use `search_knowledge_base`), or a combination of both.
2.  **Use Tools Strategically:**
    - For questions like "How has Messi been playing for Inter Miami?", you should FIRST `search_knowledge_base` to get recent news and analysis. You might then follow up with `query_database_stats` to fetch his recent stats to support the analysis.
    - For questions like "What are Gilgeous-Alexander S. points per game?", you should call `query_database_stats(player_name="Gilgeous-Alexander S.", metric="Points")`.
    - For questions like "Στατιστικά του Harden J.", you should call `query_database_stats(player_name="Harden J.", metric="all")`.
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
