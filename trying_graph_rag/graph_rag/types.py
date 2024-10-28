from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    type: str
    description: str


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    description: str
    strength: int
