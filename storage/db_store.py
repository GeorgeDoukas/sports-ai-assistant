import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.db_ingest import build_aggregates, ingest_files
from storage.db_models import Base

# ===========================================================
# Load environment & Configuration
# ===========================================================
load_dotenv()

DB_DIR = Path(os.getenv("DB_DIR", "data/db"))
DB_PATH = DB_DIR / "stats.db"
PROCESSED_STATS_FILES_LOG = DB_DIR / "processed_stats_files.log"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class DBStore:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def init_db(self):
        Base.metadata.create_all(self.engine)

    def run(self):
        self.init_db()
        with self.SessionLocal() as session:
            ingest_files(session)
            session.commit()
            build_aggregates(session)
            session.commit()
            print("Done.")


if __name__ == "__main__":
    store = DBStore()
    store.run()
