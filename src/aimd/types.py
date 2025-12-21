from pydantic import BaseModel, Field


class TextContext(BaseModel):
    """Context for text processing with title and content."""

    title: str = Field(..., description="Title of the text")
    chunk_list: list[str] = Field(..., description="List of combined text chunks")
    split_header_level: int | None = Field(
        default=None,
        description="Header level used for splitting (1-6), None if no splitting was done",
    )
