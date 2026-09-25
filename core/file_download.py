import httpx
import os
from schemas.ingest import Attachment

D360_API_KEY = os.getenv("D360_API_KEY")


import httpx
import os
import time
import logging
from schemas.ingest import Attachment

logger = logging.getLogger(__name__)

D360_API_KEY = os.getenv("D360_API_KEY")


async def resolve_attachment_bytes(attachment: Attachment) -> bytes:
    logger.info("========== MEDIA DOWNLOAD DEBUG START ==========")
    logger.info(f"Attachment raw data keys: {list(attachment.data.keys()) if attachment.data else 'none'}")

    # 1️⃣ Buffer handling — MUST run before the D360 gate below, otherwise
    # a locally-uploaded PDF (multipart /api/ingest) silently fails whenever
    # WhatsApp media isn't configured. Buffer attachments carry their own
    # bytes and don't need to hit 360dialog.
    if attachment.data.get("type") == "Buffer":
        logger.info("Attachment type: Buffer")
        data = bytes(attachment.data.get("data", []))
        logger.info(f"Buffer size: {len(data)} bytes")
        return data

    # 2️⃣ 360dialog-backed WhatsApp media download requires the API key.
    if not D360_API_KEY:
        logger.error("D360_API_KEY is NOT set — needed for WhatsApp media lookup")
        raise RuntimeError("D360_API_KEY is not set")

    logger.info(f"D360_API_KEY present: {bool(D360_API_KEY)}")

    # 2️⃣ Extract media ID
    media_id = (
        attachment.data.get("id")
        or attachment.data.get("contentId")
    )

    logger.info(f"Resolved media_id: {media_id}")

    if not media_id:
        logger.error("No media_id found in attachment")
        raise ValueError("No media_id found in attachment")

    async with httpx.AsyncClient(timeout=60.0) as client:

        # -----------------------
        # STEP 1 — GET MEDIA META
        # -----------------------
        meta_url = f"https://waba-v2.360dialog.io/{media_id}"
        logger.info(f"[STEP 1] Requesting media meta URL: {meta_url}")

        start_time = time.time()

        meta_response = await client.get(
            meta_url,
            headers={
                "D360-API-KEY": D360_API_KEY
            }
        )

        elapsed = time.time() - start_time
        logger.info(f"[STEP 1] Status: {meta_response.status_code}")
        logger.info(f"[STEP 1] Time: {elapsed:.2f}s")

        if meta_response.status_code != 200:
            logger.error(f"[STEP 1] Response body: {meta_response.text}")
            meta_response.raise_for_status()

        meta_json = meta_response.json()
        logger.info(f"[STEP 1] Response JSON: {meta_json}")

        lookaside_url = meta_json.get("url")

        if not lookaside_url:
            logger.error("No media URL returned from 360dialog")
            raise ValueError("No media URL returned from 360dialog")

        logger.info(f"[STEP 1] Lookaside URL: {lookaside_url}")

        # -----------------------
        # STEP 2 — DOWNLOAD FILE
        # -----------------------
        download_url = lookaside_url.replace(
            "https://lookaside.fbsbx.com",
            "https://waba-v2.360dialog.io"
        ).replace("\\/", "/")

        logger.info(f"[STEP 2] Download URL: {download_url}")

        start_time = time.time()

        file_response = await client.get(
            download_url,
            headers={
                "D360-API-KEY": D360_API_KEY
            }
        )

        elapsed = time.time() - start_time
        logger.info(f"[STEP 2] Status: {file_response.status_code}")
        logger.info(f"[STEP 2] Time: {elapsed:.2f}s")

        if file_response.status_code == 401:
            logger.error("401 Unauthorized during file download")
            logger.error(f"[STEP 2] Response body: {file_response.text}")
            raise Exception("401 Unauthorized from 360dialog file download")

        if file_response.status_code != 200:
            logger.error(f"[STEP 2] Response body: {file_response.text}")
            file_response.raise_for_status()

        file_size = len(file_response.content)
        logger.info(f"[STEP 2] File downloaded successfully. Size: {file_size} bytes")

        logger.info("========== MEDIA DOWNLOAD DEBUG END ==========")

        return file_response.content