from typing import List, Optional

from fastapi import APIRouter, Depends, Query, status

from src.domain.constants import ChatSessionStatus, MessageRole, ContentType
from src.domain.entities.users import UserEntityWithDetails
from src.presentation.api.schemas.requests.chat import (
    ChatSessionCreateRequest,
    ChatMessageCreateRequest,
    TriageRunCreateRequest,
    AddCandidatesRequest,
)
from src.presentation.api.schemas.responses.chat import (
    ChatSessionResponse,
    ChatSessionWithMessagesResponse,
    ChatMessageResponse,
    TriageRunResponse,
    TriageRunWithDetailsResponse,
    TriageCandidateWithDoctorResponse,
)
from src.presentation.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_chat_use_case,
    get_triage_use_case,
    get_openai_service,
    get_doctor_use_case,
)
from src.infrastructure.services.openai_service import OpenAIService
from src.use_cases.doctors.use_case import DoctorUseCase
from src.use_cases.chat.use_case import ChatUseCase
from src.use_cases.triage.use_case import TriageUseCase

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """Create a new chat session. Can be used by authenticated users or guests."""
    return await use_case.create_session(
        user_id=current_user.id if current_user else None,
        source=request.source,
        locale=request.locale,
        context_json=request.context_json,
    )


@router.get(
    "/sessions/me",
    response_model=List[ChatSessionResponse],
)
async def get_my_chat_sessions(
    session_status: Optional[ChatSessionStatus] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: UserEntityWithDetails = Depends(get_current_user),
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """Get all chat sessions for the current user."""
    return await use_case.get_my_sessions(
        current_user.id, status=session_status, skip=skip, limit=limit
    )


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionWithMessagesResponse,
)
async def get_chat_session(
    session_id: int,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """Get a chat session with all messages."""
    return await use_case.get_session_with_messages(
        session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )


@router.post(
    "/sessions/{session_id}/close",
    response_model=ChatSessionResponse,
)
async def close_chat_session(
    session_id: int,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """Close a chat session."""
    return await use_case.close_session(
        session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: int,
    request: ChatMessageCreateRequest,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: ChatUseCase = Depends(get_chat_use_case),
    openai_service: OpenAIService = Depends(get_openai_service),
    doctor_use_case: DoctorUseCase = Depends(get_doctor_use_case),
):
    """Send a message to a chat session and get AI response."""
    # Save the user message
    user_message = await use_case.send_message(
        session_id=session_id,
        content=request.content,
        role=request.role,
        content_type=request.content_type,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )

    # If user sent a message, generate AI response
    if request.role == MessageRole.USER:
        try:
            # Get all messages for context
            all_messages = await use_case.get_messages(
                session_id=session_id,
                user_id=current_user.id if current_user else None,
                is_admin=current_user.is_admin if current_user else False,
            )

            # Format messages for OpenAI
            openai_messages = [
                {"role": msg.role.value if hasattr(msg.role, 'value') else msg.role, "content": msg.content}
                for msg in all_messages
            ]

            # Get available doctors for recommendations (approved only)
            doctors = await doctor_use_case.get_all_doctors(skip=0, limit=50, is_admin=False)

            # Generate AI response
            ai_response = await openai_service.chat(
                messages=openai_messages,
                doctors=doctors,
                temperature=0.7,
            )

            # Save AI response as assistant message
            await use_case.send_message(
                session_id=session_id,
                content=ai_response,
                role=MessageRole.ASSISTANT,
                content_type=ContentType.TEXT,
                user_id=None,
                is_admin=True,  # Allow system to post
                model_name="gpt-4o-mini ",
            )
        except Exception as e:
            # Log the error but don't fail the request
            import logging
            logging.error(f"Failed to generate AI response: {e}")

    return user_message


@router.get(
    "/sessions/{session_id}/messages",
    response_model=List[ChatMessageResponse],
)
async def get_messages(
    session_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """Get all messages for a chat session."""
    return await use_case.get_messages(
        session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
        skip=skip,
        limit=limit,
    )


@router.post(
    "/sessions/{session_id}/triage",
    response_model=TriageRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_triage_run(
    session_id: int,
    request: TriageRunCreateRequest,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Create a new triage run for doctor matching."""
    return await use_case.create_triage_run(
        session_id=session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
        trigger_message_id=request.trigger_message_id,
        inputs_json=request.inputs_json,
        recommended_specialization_id=request.recommended_specialization_id,
    )


@router.get(
    "/sessions/{session_id}/triage",
    response_model=List[TriageRunResponse],
)
async def get_triage_runs(
    session_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Get all triage runs for a chat session."""
    return await use_case.get_triage_runs_by_session(
        session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/sessions/{session_id}/triage/latest",
    response_model=Optional[TriageRunWithDetailsResponse],
)
async def get_latest_triage_run(
    session_id: int,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Get the latest triage run for a chat session."""
    return await use_case.get_latest_triage_run(
        session_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )


@router.get(
    "/triage/{triage_run_id}",
    response_model=TriageRunWithDetailsResponse,
)
async def get_triage_run(
    triage_run_id: int,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Get a triage run with details."""
    return await use_case.get_triage_run_by_id(
        triage_run_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )


@router.post(
    "/triage/{triage_run_id}/candidates",
    response_model=List[TriageCandidateWithDoctorResponse],
    status_code=status.HTTP_201_CREATED,
)
async def add_candidates(
    triage_run_id: int,
    request: AddCandidatesRequest,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Add doctor candidates to a triage run."""
    candidates = [c.model_dump() for c in request.candidates]
    return await use_case.add_candidates(
        triage_run_id,
        candidates,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )


@router.get(
    "/triage/{triage_run_id}/candidates",
    response_model=List[TriageCandidateWithDoctorResponse],
)
async def get_candidates(
    triage_run_id: int,
    current_user: Optional[UserEntityWithDetails] = Depends(get_current_user_optional),
    use_case: TriageUseCase = Depends(get_triage_use_case),
):
    """Get doctor candidates for a triage run."""
    return await use_case.get_candidates(
        triage_run_id,
        user_id=current_user.id if current_user else None,
        is_admin=current_user.is_admin if current_user else False,
    )
