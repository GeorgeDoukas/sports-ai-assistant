from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from llm.llm_services import LANGUAGE, get_llm
from storage.db_store import DBStore
from storage.vector_store import VectorStoreManager

load_dotenv()

console = Console()
vs_manager = VectorStoreManager()
db_store = DBStore()


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
def query_database_stats(
    entity_name: str, scope: str = "averages", metric: str = "all", limit: int = 5
) -> str:
    """
    Use this tool to retrieve specific statistics from the database for players OR teams.

    Args:
        entity_name (str): The full name of the Player (e.g., 'Gilgeous-Alexander S.', 'Messi L.') or Team (e.g., 'Οκλαχόμα Σίτι Θάντερ').
        scope (str): Specifies the type of query. Must be one of:
            - 'averages' (default): Get a player's season averages (e.g., points/game).
            - 'team_matches': Get the team's last match results (score-wise).
            - 'player_recent': Get a player's individual performance stats for their last X games.
        metric (str): Used only with scope='averages'. The specific stat to look up (e.g., 'Points', 'Goals', 'all').
        limit (int): Used with scope='team_matches' or 'player_recent'. The number of recent games (X) to look up. Defaults to 5.
    """
    try:
        # 1. Handle Team Match History
        if scope == "team_matches":
            return db_store.get_team_last_matches(entity_name, limit)

        # 2. Handle Player Recent Performance (Last X Games)
        elif scope == "player_recent":
            return db_store.get_player_last_games(entity_name, limit)
        # 3. Handle Player Averages (Default)
        elif scope == "averages":
            return db_store.get_player_averages(entity_name, metric)

        return f"Invalid scope '{scope}'. Must be 'averages', 'team_matches', or 'player_recent'."

    except Exception as e:
        return f"An error occurred while querying the database: {type(e).__name__}"


# ==================================================
# TOOL 3: Find Ambiguous Player Names (Disambiguation)
# ==================================================
@tool
def find_ambiguous_players(surname: str) -> str:
    """
    Use this tool immediately when the user provides an ambiguous or partial name (e.g., 'Butler', 'Williams')
    to get a list of matching players, their teams, and sports.
    The output helps you ask a clarifying question to the user.
    """
    results: List[Dict[str, str]] = db_store.get_players_by_surname(surname)

    if not results:
        return f"No players found with the surname '{surname}' in the database."

    # Format the results for the LLM's consumption/response generation
    formatted_results = [
        f"Found the following {len(results)} players matching '{surname}':"
    ]
    for i, p in enumerate(results):
        formatted_results.append(
            f"{i+1}. **{p['full_name']}** (Team: {p['team']}, Sport: {p['sport']})"
        )

    return "\n".join(formatted_results)


def setup_agent():
    model = get_llm()
    current_date = datetime.now().strftime("%Y-%m-%d")
    language = LANGUAGE
    prompt = f"""
You are 'SportSense', a highly knowledgeable and data-driven sports analyst AI. Your goal is to provide insightful, accurate, and up-to-date answers to sports-related questions.
**Your final answer MUST be in the following language: {language}**
You have access to three powerful tools to help you:
1.  `search_knowledge_base`: Use this for news, analysis, and context.
2.  `query_database_stats`: Use this for hard, quantitative data (averages, match history, player recent performance).
3.  `find_ambiguous_players`: **CRITICAL!** Use this immediately when the user provides an ambiguous or partial name (e.g., "Butler", "Williams", "Messi") to get a list of options.

**Your Strategy:**
1.  **Name Clarification (NEW STEP):** If the user's query contains an ambiguous or partial player name, your FIRST step must be to call `find_ambiguous_players(surname="[surname]")`.
    - If the tool returns multiple players, you MUST ask the user a **clarifying question** (e.g., "Which Butler are you referring to? Jimmy Butler of the Heat or Malcolm Butler of the Patriots?"). Do NOT try to guess.
    - If the tool returns exactly one player, or if the name is clearly specific (e.g., "Gilgeous-Alexander S."), proceed with the query using that player's full name.
2.  **Translate/Normalize (IMPLICIT STEP):** Handle translations or normalization within your thought process if necessary, ensuring the `entity_name` passed to the tools matches the expected format in the Greek database (e.g., use 'Harden J.' not 'James Harden').
3.  **Use Tools Strategically:**
    - "How did Butler play?" -> Call `find_ambiguous_players(surname="Butler")`.
    - "What are Gilgeous-Alexander S. points per game?" -> Call `query_database_stats(entity_name="Gilgeous-Alexander S.", scope="averages", metric="points")`.
    - "How did Oklahoma City Thunder perform in their last 3 games?" -> Call `query_database_stats(entity_name="Οκλαχόμα Σίτι Θάντερ", scope="team_matches", limit=3)`.
4.  **Synthesize, Don't Just Report:** Combine information into a coherent, well-written answer in Greek, maintaining the language requested.
Today's Date is: {current_date}
Begin your thought process below to answer the user's question. Use the tools available to you.
"""
    agent = create_agent(
        model,
        system_prompt=prompt,
        tools=[search_knowledge_base, query_database_stats, find_ambiguous_players],
        checkpointer=InMemorySaver(),
    )

    return agent


# ==================================================
# 🖥️ Main Chat Loop
# ==================================================
def llm_chat():
    console.print(
        Panel(
            "⚡ Sports Insight Agent: Articles + Stats ⚡",
            border_style="blue",
        )
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
