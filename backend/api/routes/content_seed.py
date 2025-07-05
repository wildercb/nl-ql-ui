from fastapi import APIRouter, Depends, HTTPException

from services.content_seed_service import get_seed_service, ContentSeedService

router = APIRouter(prefix="/api", tags=["Data Seed"])


@router.post("/data/seed")
async def seed_demo_data(force: bool = False, service: ContentSeedService = Depends(get_seed_service)):
    """Populate the content database with demo documents. Does nothing if data already present."""
    try:
        inserted = await service.seed_once(force=force)
        return {"inserted": inserted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) 