import os
import argparse
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import feedparser

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM backends
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

# === RSS FEEDS (Greek sports) ===
RSS_FEEDS = {
    'super-league': 'https://www.sport24.gr/rss/football/super-league',
    'greek-basket': 'https://www.sport24.gr/rss/basketball/greek-basket-league',
    'euroleague': 'https://www.sport24.gr/rss/basketball/euroleague',
    'champions-league': 'https://www.sport24.gr/rss/football/champions-league',
    'nba': 'https://www.sport24.gr/rss/basketball/nba',
}

# === Sport/League to RSS mapper ===
SPORT_LEAGUE_TO_RSS = {
    ('football', 'super-league'): ['super-league'],
    ('football', 'champions-league'): ['champions-league'],
    ('basketball', 'greek-basket'): ['greek-basket'],
    ('basketball', 'euroleague'): ['euroleague'],
    ('basketball', 'nba'): ['nba'],
    ('football', None): ['super-league', 'champions-league'],
    ('basketball', None): ['greek-basket', 'euroleague', 'nba'],
}

def fetch_greek_news(sport: str, league: str = None, days_back: int = 2) -> str:
    """Fetch recent Greek sports news from Sport24 RSS feeds."""
    keys = SPORT_LEAGUE_TO_RSS.get((sport.lower(), league.lower() if league else None))
    if not keys:
        keys = SPORT_LEAGUE_TO_RSS.get((sport.lower(), None), [])
    
    if not keys:
        return "No matching RSS feeds found for this sport."

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    for key in keys:
        if key not in RSS_FEEDS:
            continue
        print(f"📡 Fetching RSS: {key}")
        try:
            feed = feedparser.parse(RSS_FEEDS[key])
            for entry in feed.entries[:5]:  # Top 5 per feed
                pub_date = datetime(*entry.published_parsed[:6]) if entry.published_parsed else datetime.min
                if pub_date > cutoff:
                    articles.append(f"- {entry.title} ({key})")
        except Exception as e:
            print(f"⚠️ RSS error for {key}: {e}")

    return "\n".join(articles[:15]) if articles else "No recent Greek sports news."

def get_stoiximan_odds(sport: str, league: str = None, match_filter: str = None):
    """Fetch Stoiximan odds via The Odds API."""
    # Map to OddsAPI sport keys
    ODDSAPI_SPORTS = {
        'football': 'soccer_greece_super_league',
        'basketball': 'basketball_greece_hea',
    }
    # Override for specific leagues
    LEAGUE_TO_ODDSAPI = {
        'super-league': 'soccer_greece_super_league',
        'champions-league': 'soccer_uefa_champs_league',
        'euroleague': 'basketball_euroleague',
        'nba': 'basketball_nba',
        'greek-basket': 'basketball_greece_hea',
    }

    market_key = LEAGUE_TO_ODDSAPI.get(league.lower()) if league else ODDSAPI_SPORTS.get(sport.lower())
    if not market_key:
        return "No matching OddsAPI market found."

    url = f"https://api.theoddsapi.com/v4/sports/{market_key}/odds"
    params = {
        'apiKey': os.getenv('ODDSAPI_KEY'),
        'bookmakers': 'stoiximan',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal'
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return f"OddsAPI error: {res.text}"
        data = res.json()
        if not isinstance(data, list):
            return "No odds data returned."

        # Filter by match if given
        if match_filter:
            filtered = []
            for event in data:
                if (match_filter.lower() in event['home_team'].lower() or
                    match_filter.lower() in event['away_team'].lower()):
                    filtered.append(event)
            data = filtered

        odds_lines = []
        for event in data[:3]:
            home, away = event['home_team'], event['away_team']
            bookmakers = event.get('bookmakers', [])
            stoiximan = next((b for b in bookmakers if b['key'] == 'stoiximan'), None)
            if not stoiximan:
                continue
            h2h = next((m for m in stoiximan['markets'] if m['key'] == 'h2h'), None)
            if h2h:
                outcomes = ", ".join([f"{o['name']}: {o['price']}" for o in h2h['outcomes']])
                odds_lines.append(f"{home} vs {away} → {outcomes}")
        return "\n".join(odds_lines) if odds_lines else "No Stoiximan odds found."
    except Exception as e:
        return f"Error fetching odds: {e}"

def get_llm(backend: str, model: str):
    """Return LangChain LLM instance."""
    if backend == "ollama":
        return ChatOllama(model=model, temperature=0.3, num_predict=500)
    elif backend == "llmstudio":
        base_url = os.getenv("LLM_STUDIO_URL", "http://localhost:1234/v1")
        return ChatOpenAI(
            base_url=base_url,
            api_key="not-needed",  # LLM Studio ignores this
            model=model,
            temperature=0.3,
            max_tokens=500
        )
    else:
        raise ValueError("Backend must be 'ollama' or 'llmstudio'")

def main():
    parser = argparse.ArgumentParser(description="Local LLM Betting Suggester (LangChain + Greek RSS)")
    parser.add_argument('--sport', required=True, choices=['football', 'basketball'], help="Sport")
    parser.add_argument('--league', help="League (e.g., super-league, euroleague, nba, champions-league, greek-basket)")
    parser.add_argument('--match', help="Specific match to focus on")
    parser.add_argument('--llm-backend', choices=['ollama', 'llmstudio'], default='ollama')
    parser.add_argument('--llm-model', default='llama3')
    args = parser.parse_args()

    # 1. Fetch Greek news
    print("📰 Fetching Greek sports news...")
    news = fetch_greek_news(args.sport, args.league)

    # 2. Fetch Stoiximan odds
    print("💰 Fetching Stoiximan odds...")
    odds = get_stoiximan_odds(args.sport, args.league, args.match)

    # 3. Build prompt
    prompt_template = """
You are a sharp Greek sports betting analyst. Analyze the following and propose 1-2 value bets on Stoiximan.

Recent Greek/international news:
{news}

Stoiximan odds:
{odds}

Guidelines:
- Only suggest if there's a clear mismatch (e.g., injury, motivation, odds lag).
- Mention specific teams/players if relevant.
- If a match is specified, focus on it.
- Be concise, factual, and cautious.

Output format:
**Suggested Bet**: [type]
**Odds**: [value]
**Reasoning**: [2-3 sentences]
**Confidence**: Low / Medium / High
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    if args.match:
        prompt_template += f"\nFocus match: {args.match}"
        prompt = ChatPromptTemplate.from_template(prompt_template)

    # 4. Run LLM
    print(f"🧠 Running {args.llm_backend} ({args.llm_model})...")
    llm = get_llm(args.llm_backend, args.llm_model)
    chain = prompt | llm | StrOutputParser()

    try:
        suggestion = chain.invoke({"news": news, "odds": odds})
    except Exception as e:
        suggestion = f"LLM error: {e}"

    # 5. Output
    print("\n" + "="*70)
    print("💡 BETTING SUGGESTION (Greek Focus)")
    print("="*70)
    print(suggestion)
    print("\n📊 Stoiximan Odds:")
    print(odds if odds else "None")
    print("\n⚠️  DISCLAIMER: For informational use only. Gambling is risky. Bet responsibly.")
    print("="*70)

if __name__ == "__main__":
    if not os.getenv("ODDSAPI_KEY"):
        print("❌ Missing ODDSAPI_KEY in .env")
        exit(1)
    main()