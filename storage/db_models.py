from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Sport(Base):
    __tablename__ = "sports"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    competitions = relationship("Competition", back_populates="sport")
    teams = relationship("Team", back_populates="sport")


class Competition(Base):
    __tablename__ = "competitions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    sport_id = Column(Integer, ForeignKey("sports.id"), nullable=False)

    sport = relationship("Sport", back_populates="competitions")
    teams = relationship("Team", back_populates="competition")
    matches = relationship("Match", back_populates="competition")

    __table_args__ = (UniqueConstraint("name", "sport_id"),)


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    sport_id = Column(Integer, ForeignKey("sports.id"), nullable=False)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)

    sport = relationship("Sport", back_populates="teams")
    competition = relationship("Competition", back_populates="teams")
    players = relationship("Player", back_populates="team")

    __table_args__ = (UniqueConstraint("name", "sport_id", "competition_id"),)


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    team = relationship("Team", back_populates="players")
    football_stats = relationship("FootballStats", back_populates="player")
    basketball_stats = relationship("BasketballStats", back_populates="player")

    __table_args__ = (UniqueConstraint("name", "team_id"),)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    sport_id = Column(Integer, ForeignKey("sports.id"), nullable=False)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)

    competition = relationship("Competition", back_populates="matches")
    football_stats = relationship("FootballStats", back_populates="match")
    basketball_stats = relationship("BasketballStats", back_populates="match")

    __table_args__ = (
        UniqueConstraint("date", "home_team_id", "away_team_id", "competition_id"),
    )


class FootballStats(Base):
    __tablename__ = "football_stats"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    rating = Column(Float)
    shots = Column(Float)
    xg = Column(Float)
    passes = Column(String)
    touches = Column(Float)
    touches_box = Column(Float)
    dribbles = Column(String)
    duels = Column(Float)
    position = Column(String)

    match = relationship("Match", back_populates="football_stats")
    player = relationship("Player", back_populates="football_stats")


class FootballPlayerTotals(Base):
    __tablename__ = "football_player_totals"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True, nullable=False)

    games = Column(Integer, nullable=False, default=0)
    rating = Column(Float)
    shots = Column(Float)
    xg = Column(Float)
    touches = Column(Float)
    touches_box = Column(Float)
    duels = Column(Float)


class FootballPlayerPerGame(Base):
    __tablename__ = "football_player_pergame"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True, nullable=False)

    games = Column(Integer, nullable=False, default=0)
    rating = Column(Float)
    shots = Column(Float)
    xg = Column(Float)
    touches = Column(Float)
    touches_box = Column(Float)
    duels = Column(Float)


class BasketballStats(Base):
    __tablename__ = "basketball_stats"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    points = Column(Float)
    rebounds_total = Column(Float)
    assists = Column(Float)
    minutes = Column(Float)
    fg_made = Column(Float)
    fg_attempts = Column(Float)
    two_made = Column(Float)
    two_attempts = Column(Float)
    three_made = Column(Float)
    three_attempts = Column(Float)
    ft_made = Column(Float)
    ft_attempts = Column(Float)
    plus_minus = Column(Integer)
    off_rebounds = Column(Float)
    def_rebounds = Column(Float)
    fouls = Column(Float)
    steals = Column(Float)
    turnovers = Column(Float)
    blocks = Column(Float)
    blocks_against = Column(Float)
    tech_fouls = Column(Float)

    match = relationship("Match", back_populates="basketball_stats")
    player = relationship("Player", back_populates="basketball_stats")


class BasketballPlayerTotals(Base):
    __tablename__ = "basketball_player_totals"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True, nullable=False)

    games = Column(Integer, nullable=False, default=0)
    points = Column(Float)
    rebounds = Column(Float)
    assists = Column(Float)
    steals = Column(Float)
    blocks = Column(Float)
    turnovers = Column(Float)
    minutes = Column(Float)


class BasketballPlayerPerGame(Base):
    __tablename__ = "basketball_player_pergame"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True, nullable=False)

    games = Column(Integer, nullable=False, default=0)
    points = Column(Float)
    rebounds = Column(Float)
    assists = Column(Float)
    steals = Column(Float)
    blocks = Column(Float)
    turnovers = Column(Float)
    minutes = Column(Float)
