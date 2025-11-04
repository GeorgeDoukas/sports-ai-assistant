from pathlib import Path

from storage.db_ingest import ingest_files, build_aggregates
from storage.db_models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_PATH = Path("data/db/stats.db")
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
