from pydantic import BaseModel
from typing import Optional, List

class Attachment(BaseModel):
    filename: str
    url: str
    content_type: str

class IngestRequest(BaseModel):
    source_channel: str  # email | whatsapp
    source_message_id: str
    source_filename: str
    supplier_email: Optional[str]
    supplier_name: Optional[str]
    attachments: List[Attachment]
    text_body: Optional[str]
