import argparse
import os
import subprocess
import sys
from pathlib import Path

import dotenv
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt  # For user input
from rich.syntax import Syntax  # For syntax highlighting if needed
from rich.text import Text  # For styled text

console = Console()

# --- Import your existing modules ---
# You will need to ensure these modules are importable from this script's location.
try:
    from llm.generate_daily_reports import ReportGenerator
    from llm.llm_chat import llm_chat
    from llm.process_articles import ArticleProcessor
    from scrapers.sports_news_scraper import scrape_news
    from scrapers.stats_scraper import scrape_stats
    from storage.db_store import DBStore
    from storage.vector_store import VectorStoreManager
except ImportError as e:
    console.print(f"[red]❌ Failed to import required modules: {e}[/red]")
    console.print(
        "Make sure all your modules are accessible and dependencies are installed."
    )
    sys.exit(1)


def display_main_menu():
    """Display the main menu and get user choice."""
    console.clear()
    console.print("\n[bold blue]Sports News Pipeline CLI[/bold blue]")
    console.print("1. Configuration")
    console.print("2. Individual Modules")
    console.print("3. Get News And Stats")
    console.print("4. Summarize Articles And Generate Daily Reports")
    console.print("5. Open Chat")
    console.print("6. Quit")
    choice = IntPrompt.ask(
        "Select an option", choices=[str(i) for i in range(1, 7)], show_choices=False
    )
    return choice


def display_generate_reports_menu():
    console.clear()
    console.print("\n[bold blue]Generate Reports Options:[/bold blue]")
    console.print("1. Generate All")
    console.print("2. Generate Specific Date")
    console.print("3. Back")
    gr_choice = IntPrompt.ask(
        "Select an option",
        choices=[str(i) for i in range(1, 4)],
        show_choices=False,
    )

    return gr_choice


def display_process_articles_menu():
    console.clear()
    console.print("\n[bold blue]Process Articles Options:[/bold blue]")
    console.print("1. Process Specific")
    console.print("2. Process All")
    console.print("3. Back")
    pa_choice = IntPrompt.ask(
        "Select an option",
        choices=[str(i) for i in range(1, 4)],
        show_choices=False,
    )

    return pa_choice


def display_vectorstore_menu():
    console.clear()
    console.print("\n[bold blue]VectorStore Options:[/bold blue]")
    console.print("1. Create/Update")
    console.print("2. Query")
    console.print("3. Clear")
    console.print("4. Back")
    vs_choice = IntPrompt.ask(
        "Select an option",
        choices=[str(i) for i in range(1, 5)],
        show_choices=False,
    )

    return vs_choice


def display_individual_modules_menu():
    console.clear()
    console.print("\n[bold blue]Individual Modules:[/bold blue]")
    console.print("1. DBStore")
    console.print("2. VectorStore")
    console.print("3. Stats Scraper")
    console.print("4. News Scraper")
    console.print("5. Process Articles")
    console.print("6. Generate Reports")
    console.print("7. Back")
    ind_choice = IntPrompt.ask(
        "Select an option",
        choices=[str(i) for i in range(1, 8)],
        show_choices=False,
    )

    return ind_choice


def load_config_file() -> Path:
    """Load the .env file path."""
    config_path = Path(".env")
    if not config_path.exists():
        console.print(
            f"[yellow]⚠️  Config file {config_path} not found. Creating a default one...[/yellow]"
        )

    return config_path


def edit_config_file():
    """Open the .env file in the system editor for editing."""
    config_path = load_config_file()
    console.print(f"[bold blue]Editing configuration file: {config_path}[/bold blue]")
    console.print(
        "Make your changes in the editor that opens now. Save and close the editor to return."
    )

    editor = os.environ.get(
        "EDITOR", "nano"
    )  # Use $EDITOR if set, otherwise default to 'nano'
    if os.name == "nt":  # Windows
        editor = os.environ.get(
            "EDITOR", "notepad"
        )  # Use $EDITOR if set, otherwise default to 'notepad'

    try:
        # Use subprocess to open the editor
        subprocess.run([editor, str(config_path)], check=True)
        console.print("\n[green]✅ Configuration file saved.[/green]")
        # Reload environment variables after editing
        dotenv.load_dotenv(dotenv_path=config_path, override=True)
        console.print("[green]✅ Environment variables reloaded.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]❌ Error opening editor or saving file: {e}[/red]")
    except FileNotFoundError:
        console.print(
            f"\n[red]❌ Editor '{editor}' not found. Please set the EDITOR environment variable to your preferred editor.[/red]"
        )
        console.print(f"Example (Linux/Mac): export EDITOR=nano")
        console.print(f"Example (Windows): set EDITOR=notepad")


def run_dbstore():
    console.clear()
    console.print("--- Running DBStore ---")
    try:
        store = DBStore()
        store.run()
        console.print("--- DBStore Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in DBStore: {e}\n")
    input("Press Enter to continue...")


def run_vectorstore_create_update():
    console.clear()
    console.print("--- Creating/Updating Vector Store ---")
    try:
        manager = VectorStoreManager()
        manager.sync()
        manager.create_or_update(days_back=30)
        console.print("--- VectorStore Create/Update Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in VectorStore Create/Update: {e}\n")
    input("Press Enter to continue...")


def run_vectorstore_query():
    console.clear()
    console.print("--- Querying VectorStore ---")
    query = Prompt.ask("[bold blue]Query[/bold blue]")
    try:
        manager = VectorStoreManager()
        search_results = manager.query(query, k=3)
        console.print("\n" + "=" * 50)
        console.print("Query Results:")
        console.print("=" * 50)
        if search_results:
            for i, result in enumerate(search_results):
                console.print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
                console.print(f"  Title: {result['metadata'].get('title', 'N/A')}")
                console.print(f"  Sport: {result['metadata'].get('sport', 'N/A')}")
                console.print(
                    f"  Competition: {result['metadata'].get('competition', 'N/A')}"
                )
                console.print(f"  Source: {result['metadata'].get('source', 'N/A')}")
                console.print(f"  URL: {result['metadata'].get('url', 'N/A')}")
                # Truncate content for readability
                content_snippet = result["content"][:300].strip()
                console.print(f"  Content Snippet: {content_snippet}...")
        else:
            console.print("\n🤷 No results found for the query.")
        console.print("\n" + "=" * 50)
    except Exception as e:
        console.print(f"❌ Error in VectorStore Query: {e}\n")
    input("Press Enter to continue...")


def run_vectorstore_clear():
    console.clear()
    console.print("--- Clearing VectorStore ---")
    try:
        manager = VectorStoreManager()
        manager.clear()
        console.print("--- VectorStore Cleared ---\n")
    except Exception as e:
        console.print(f"❌ Error clearing VectorStore: {e}\n")
    input("Press Enter to continue...")


def run_stats_scraper():
    console.clear()
    console.print("--- Scraping Stats ---")
    scrape_stats()
    input("Press Enter to continue...")


def run_news_scraper():
    console.clear()
    console.print("--- Scraping News ---")
    scrape_news()
    input("Press Enter to continue...")


def run_process_specific_articles():
    console.clear()
    console.print("--- Processing Specific Article ---")
    file = Path(Prompt.ask("[bold blue]File Name[/bold blue]"))
    try:
        processor = ArticleProcessor(language=os.getenv("LANGUAGE", "greek"))
        if file.exists():
            processor.evaluate_single_file(file)
        else:
            print(f"❌ Error: File not found at '{file}'")
        console.print("--- Process Article Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Process Article: {e}\n")
    input("Press Enter to continue...")


def run_process_all_articles():
    console.clear()
    console.print("--- Processing All Articles ---")
    try:
        processor = ArticleProcessor(language=os.getenv("LANGUAGE", "greek"))
        processor.process_all_articles_in_parallel()
        console.print("--- Process All Articles Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Process Articles: {e}\n")
    input("Press Enter to continue...")


def run_generate_reports_all():
    console.clear()
    console.print("--- Generating All Reports ---")
    try:
        args = argparse.Namespace(
            all=True, date=None, sport=None, competition=None, method="summaries"
        )
        generator = ReportGenerator(args)
        generator.run()
        console.print("--- Generate All Reports Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Generate Reports: {e}\n")
    input("Press Enter to continue...")


def run_generate_reports_date():
    console.clear()
    specific_date = "03-Νοεμβρίου-2025"  # Example date
    console.print("--- Generating Report for Specific Date ---")
    specific_date = Prompt.ask("[bold blue]Specify Date[/bold blue]")
    try:
        args = argparse.Namespace(
            all=False,
            date=specific_date,
            sport=None,
            competition=None,
            method="summaries",
        )
        generator = ReportGenerator(args)
        generator.run()
        console.print(f"--- Generate Report for {specific_date} Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Generate Reports: {e}\n")
    input("Press Enter to continue...")


def run_get_news_workflow():
    console.clear()
    console.print("--- Starting Full 'Get News' Workflow ---")
    try:
        scrape_news()
        scrape_stats()

        # 3. Update VectorStore
        console.print("\n--- Updating VectorStore ---")
        try:
            manager = VectorStoreManager()
            manager.create_or_update(days_back=30)
            console.print("✅ VectorStore updated.")
        except Exception as e:
            console.print(f"❌ Error updating VectorStore: {e}")

        # 4. Run DB Ingest (Assuming DBStore.run() handles ingestion)
        console.print("\n--- Running DB Ingest (via DBStore) ---")
        try:
            store = DBStore()
            store.run()
            console.print("✅ DB Ingest completed.")
        except Exception as e:
            console.print(f"❌ Error in DB Ingest: {e}")

        console.print("\n" + "=" * 50)
        console.print("✅ Full 'Get News' Workflow completed.")
        console.print("=" * 50 + "\n")
    except Exception as e:
        console.print(f"\n❌ Critical error in 'Get News' workflow: {e}\n")
    console.print("--- 'Get News' Workflow Completed ---\n")
    input("Press Enter to continue...")


def summarize_articles_workflow():
    console.clear()
    console.print("--- Processing All Articles ---")
    try:
        processor = ArticleProcessor(language=os.getenv("LANGUAGE", "greek"))
        processor.process_all_articles_in_parallel()
        console.print("--- Process All Articles Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Process Articles: {e}\n")
    console.print("--- Generating All Reports ---")
    try:
        args = argparse.Namespace(
            all=True, date=None, sport=None, competition=None, method="summaries"
        )
        generator = ReportGenerator(args)
        generator.run()
        console.print("--- Generate All Reports Completed ---\n")
    except Exception as e:
        console.print(f"❌ Error in Generate Reports: {e}\n")


def run_llm_chat():
    console.print("--- Starting LLM Chat ---")
    try:
        llm_chat()
    except Exception as e:
        console.print(f"\n❌ Critical error in Starting LLM Chat: {e}\n")


def main():
    while True:
        choice = display_main_menu()
        if choice == 1:  # Configuration
            edit_config_file()  # This function handles the editing using the system editor
        elif choice == 2:  # Individual Modules
            ind_choice = display_individual_modules_menu()
            if ind_choice == 1:  # DBStore
                run_dbstore()
            elif ind_choice == 2:  # VectorStore
                vs_choice = display_vectorstore_menu()
                if vs_choice == 1:  # Create/Update
                    run_vectorstore_create_update()
                elif vs_choice == 2:  # Query
                    run_vectorstore_query()
                elif vs_choice == 3:  # Clear
                    run_vectorstore_clear()
                elif choice == 4:  # Back
                    continue
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            elif ind_choice == 3:  # Stats Scraper
                run_stats_scraper()
            elif ind_choice == 4:  # News Scraper
                run_news_scraper()
            elif ind_choice == 5:  # Process Articles
                pa_choice = display_process_articles_menu()
                if pa_choice == 1:  # Process Specific
                    run_process_specific_articles()
                elif pa_choice == 2:  # Process All
                    run_process_all_articles()
                elif choice == 3:  # Back
                    continue
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            elif ind_choice == 6:  # Generate Reports
                gr_choice = display_generate_reports_menu()
                if gr_choice == 1:  # Generate All
                    run_generate_reports_all()
                elif gr_choice == 2:  # Generate Specific Date
                    run_generate_reports_date()
                elif choice == 3:  # Back
                    continue
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            elif ind_choice == 7:  # Back
                continue
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        elif choice == 3:  # Get News And Stats
            run_get_news_workflow()
        elif choice == 4:  # Summarize Articles And Generate Daily Reports
            summarize_articles_workflow()
        elif choice == 5:  # Open Chat
            console.clear()
            run_llm_chat()
        elif choice == 6:  # Quit
            console.print("[bold red]Quitting...[/bold red]")
            break
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")


if __name__ == "__main__":
    main()
