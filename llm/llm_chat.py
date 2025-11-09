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
from llm.process_queries import translate_name as helper_translate_name, improve_vector_query as helper_improve_query
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


# ==================================================
# TOOL 4: Translate Name (Intermediate Step)
# ==================================================
@tool
def translate_name(text: str, target_lang: str) -> str:
    """
    Use this tool to translate a name (e.g., 'LeBron James') or entity to the target language 
    (obtained from the LANGUAGE environment variable, e.g., 'Greek') when an initial 
    database query fails, allowing a retry with the translated name.
    """
    # Calls the actual LLM-powered helper function
    return helper_translate_name(text)


# ==================================================
# TOOL 5: Improve Vector Store Query
# ==================================================
@tool
def improve_vector_query(original_query: str) -> str:
    """
    Use this to refine a conversational or ambiguous natural language query into a concise, 
    optimized set of keywords for better retrieval from the knowledge base (Tool 1).
    Example: 'What is the news on Messi's last game?' -> 'Messi last game news summary'
    """
    # Calls the actual LLM-powered helper function
    return helper_improve_query(original_query)

def setup_agent():
    model = get_llm()
    current_date = datetime.now().strftime("%Y-%m-%d")
    language = LANGUAGE
    prompt = f"""
You are 'SportSense', a highly knowledgeable and data-driven sports analyst AI. Your goal is to provide insightful, accurate, and up-to-date answers to sports-related questions.
**Your final answer MUST be in the following language: {language}**
You have access to five powerful tools to help you:
1.  `search_knowledge_base`: Use this for news, analysis, and context.
2.  `query_database_stats`: Use this for hard, quantitative data (averages, match history, player recent performance).
3.  `find_ambiguous_players`: **CRITICAL!** Use this immediately when the user provides an ambiguous or partial name to get a list of options.
4.  `translate_name`: Use this as a **secondary strategy** if a primary database lookup fails, to translate the name to the target language and retry.
5.  `improve_vector_query`: Use this to refine a conversational query before searching the knowledge base (Tool 1).

**Your Strategy:**
1.  **Disambiguation (Name Clarification):** If the user's query contains an ambiguous or partial player name (e.g., 'Butler'), your FIRST step must be to call `find_ambiguous_players(surname="[surname]")`.
    - If the tool returns multiple players, you MUST ask the user a **clarifying question**. Do NOT guess.
    - If the tool returns exactly one player, proceed with the query using that player's full name.

2.  **Query Improvement (News/Context):** If the request requires the knowledge base (Tool 1) and the query is conversational or vague, FIRST use `improve_vector_query(original_query)` to optimize the search keywords, and THEN use `search_knowledge_base` with the improved query.

3.  **Database Retrieval with Translation Fallback:**
    a. Try `query_database_stats` with the entity name as initially given or clarified.
    b. If the query in step (a) fails (returns "Could not find a player matching..."), use `translate_name(text=original_name)` to get the translated name.
    c. Retry `query_database_stats` with the translated result from step (b).

4.  **Synthesize, Don't Just Report:** Do not just dump the raw output from the tools. Combine the information into a coherent, well-written answer in Greek, maintaining the language requested.

**Target Language for Output/Translation:** {language}
Today's Date is: {current_date}

Begin your thought process below to answer the user's question. Use the tools available to you.
"""
    agent = create_agent(
        model,
        system_prompt=prompt,
        tools=[search_knowledge_base, query_database_stats, find_ambiguous_players, translate_name, improve_vector_query],
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
