import httpx
import os
from schemas.ingest import Attachment

D360_API_KEY = os.getenv("D360_API_KEY")


async def resolve_attachment_bytes(attachment: Attachment) -> bytes:
    if attachment.data.get("type") == "Buffer":
        return bytes(attachment.data.get("data", []))

    media_id = (
        attachment.data.get("id")
        or attachment.data.get("contentId")
    )

    if media_id:
        async with httpx.AsyncClient(timeout=60.0) as client:

            meta_response = await client.get(
                f"https://waba-v2.360dialog.io/{media_id}",
                headers={
                    "D360-API-KEY": D360_API_KEY
                }
            )
            meta_response.raise_for_status()

            meta_json = meta_response.json()
            fresh_url = meta_json.get("url")

            if not fresh_url:
                raise ValueError("360dialog did not return media URL")

            file_response = await client.get(
                fresh_url,
                headers={
                    "D360-API-KEY": D360_API_KEY
                }
            )
            file_response.raise_for_status()

            return file_response.content

    raise ValueError("Unsupported attachment format")