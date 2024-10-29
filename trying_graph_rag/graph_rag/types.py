from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    type: str
    description: str


class SummarizedUniqueEntity(BaseModel):
    name: str
    type: str
    summary: str


class Relationship(BaseModel):
    source_entity: str  # entity name
    target_entity: str
    description: str
    strength: int


class Findings(BaseModel):
    summary: str
    explanation: str


class CommunityReport(BaseModel):
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: list[Findings]


class SummarizedCommunity(BaseModel):
    community_id: int
    hierachy_level: int
    community_report: CommunityReport
