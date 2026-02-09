from typing import List, Optional, Union
from pydantic import BaseModel, Field
from pydantic.v1 import validator


class Attachment(BaseModel):
    fileName: str
    contentType: str
    checksum: str
    contentId: Optional[str] = None
    fileSize: int
    data: dict


class IngestRequest(BaseModel):
    source_channel: str
    source_message_id: str
    source_filename: str
    supplier_email: str
    supplier_name: str
    sender_email: Optional[str] = None
    sender_name: Optional[str] = None
    subject: Optional[str] = None
    text_body: Optional[str] = None
    attachments: Optional[Union[List[Attachment], Attachment]] = Field(default_factory=list)

    @validator('attachments')
    def validate_attachments(cls, v):
        if v is None:
            return []
        if isinstance(v, dict):  # Single attachment object
            return [v]
        return v