from fastapi import APIRouter, BackgroundTasks
from schemas.ingest import IngestRequest
from schemas.output import OfferResponse
from workers.processor import process_offer

router = APIRouter()

@router.post("/ingest", response_model=OfferResponse)
async def ingest(payload: IngestRequest, background_tasks: BackgroundTasks):

    offer = await process_offer(payload)

    return {
        "data": [offer]
    }
