import httpx
import os
from schemas.ingest import Attachment

D360_API_KEY = os.getenv("D360_API_KEY")


async def resolve_attachment_bytes(attachment: Attachment) -> bytes:
    if attachment.data.get("type") == "Buffer":
        return bytes(attachment.data.get("data", []))

    if "url" in attachment.data:
        media_url = attachment.data["url"]

        media_url = media_url.replace(
            "https://lookaside.fbsbx.com",
            "https://waba-v2.360dialog.io"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                media_url,
                headers={
                    "Authorization": f"Bearer {D360_API_KEY}",
                    "User-Agent": "curl/7.64.1"
                }
            )
            response.raise_for_status()
            return response.content

    raise ValueError("Unsupported attachment format")
