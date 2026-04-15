from pydantic import BaseModel, Field


class Performer(BaseModel):
    player: str
    pts: int
    reb: int
    ast: int


class GameDocument(BaseModel):
    game_id: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    top_performers: list[Performer]
    recap_text: str
    stat_summary: str
    full_text: str
