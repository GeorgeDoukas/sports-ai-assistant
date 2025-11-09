from datetime import datetime

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


def setup_agent():
    model = get_llm()
    current_date = datetime.now().strftime("%Y-%m-%d")
    language = LANGUAGE
    prompt = f"""
You are 'SportSense', a highly knowledgeable and data-driven sports analyst AI. Your goal is to provide insightful, accurate, and up-to-date answers to sports-related questions.
**Your final answer MUST be in the following language: {language}**
You have access to two powerful tools to help you:
1.  `search_knowledge_base`: Use this to get the latest news, expert analysis, commentary, and context on players, teams, and events.
2.  `query_database_stats`: Use this to get hard, quantitative data and statistics for **players OR teams**.
    **CRITICAL:** This tool now supports three query scopes, specified by the `scope` argument:
    1.  `scope='averages'` (Default): For a player's season averages (e.g., points/game, goals/game).
        - **Arguments**: `entity_name` (Player Name), `metric` (e.g., 'Points', 'Goals', 'all').
    2.  `scope='team_matches'`: For a team's last match results (score-wise).
        - **Arguments**: `entity_name` (Team Name), `limit` (number of games, default 5).
    3.  `scope='player_recent'`: For a player's individual performance stats in their last X games.
        - **Arguments**: `entity_name` (Player Name), `limit` (number of games, default 5).
**Your Strategy:**
1.  **Analyze the User's Query:** Understand if the user is asking for:
    - Objective **Averages/Totals** (Use `query_database_stats` with `scope='averages'`).
    - **Team Match History** (Use `query_database_stats` with `scope='team_matches'`).
    - **Player Game-by-Game Performance** (Use `query_database_stats` with `scope='player_recent'`).
    - Subjective analysis/news (Use `search_knowledge_base`).
2.  **Use Tools Strategically:**
    - "Πόσους πόντους έχει ο Gilgeous-Alexander S. ανά παιχνίδι;": `query_database_stats(entity_name="Gilgeous-Alexander S.", scope="averages", metric="points")`
    - "Πώς πήγε η Οκλαχόμα Σίτι Θάντερ στα 3 τελευταία ματς;": `query_database_stats(entity_name="Οκλαχόμα Σίτι Θάντερ", scope="team_matches", limit=3)`
    - "Πώς έπαιξε ο Harden J. στους 5 τελευταίους αγώνες;": `query_database_stats(entity_name="Harden J.", scope="player_recent", limit=5)`
3.  **Synthesize, Don't Just Report:** Do not just dump the raw output from the tools. Combine the information into a coherent, well-written answer in Greek.
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
