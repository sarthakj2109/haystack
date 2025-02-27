from fastapi import APIRouter

from rest_api.controller import file_upload, search, feedback,documents

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(feedback.router, tags=["feedback"])
router.include_router(file_upload.router, tags=["file-upload"])
router.include_router(documents.router, tags=["documents"])