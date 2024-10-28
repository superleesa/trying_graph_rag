from pydantic import BaseModel


class Entity(BaseModel):
    entity_name: str
    entity_type: str
    entity_description: str


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_strength: int
