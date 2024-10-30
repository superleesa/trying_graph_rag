from pydantic import BaseModel, AliasChoices, Field


class Entity(BaseModel):
    name: str = Field(validation_alias=AliasChoices("entity_name", "name"))
    type: str = Field(validation_alias=AliasChoices("entity_type", "type"))
    description: str = Field(validation_alias=AliasChoices("entity_description", "description"))


class SummarizedUniqueEntity(BaseModel):
    name: str
    type: str
    summary: str


class Relationship(BaseModel):
    source_entity: str  # entity name
    target_entity: str
    description: str = Field(validation_alias=AliasChoices("relationship_description", "description"))
    strength: int = Field(validation_alias=AliasChoices("relationship_strength", "strength"))


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


class GraphIndex(BaseModel):
    all_entities: list[Entity]
    all_relationships: list[Relationship]
    unique_entities: list[SummarizedUniqueEntity]
    hierachical_communities: dict[int, list[SummarizedCommunity]]


class RelevantPointToQuery(BaseModel):
    description: str
    score: float  # should be within 0 - 100
