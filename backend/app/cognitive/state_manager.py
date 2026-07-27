"""Persistence for the live cognitive runtime."""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.cognitive.mind_state import MindState
from app.cognitive.prompt_builder import PromptPlan
from app.onboarding.profile_schema import UserProfile


class RuntimeSnapshot(BaseModel):
    schema_version: int = 1
    hearth_version: str = "phase0"
    model_version: str = "unknown"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profile: UserProfile
    mind_state: MindState
    conversation_state: dict = Field(default_factory=dict)
    last_prompt_plan: PromptPlan | None = None
    runtime_metrics: dict = Field(default_factory=dict)


class StateManager:
    def __init__(self, snapshot_path: Path, hearth_version: str, model_version: str):
        self.snapshot_path = snapshot_path
        self.hearth_version = hearth_version
        self.model_version = model_version

    def create_mind_state(self) -> MindState:
        return MindState()

    def load_snapshot(self) -> RuntimeSnapshot | None:
        if not self.snapshot_path.exists():
            return None
        return RuntimeSnapshot.model_validate_json(self.snapshot_path.read_text(encoding="utf-8"))

    def save_snapshot(self, profile: UserProfile, mind_state: MindState, conversation_state: dict, last_prompt_plan: PromptPlan | None, runtime_metrics: dict | None = None) -> RuntimeSnapshot:
        snapshot = RuntimeSnapshot(
            hearth_version=self.hearth_version,
            model_version=self.model_version,
            profile=profile,
            mind_state=mind_state,
            conversation_state=conversation_state,
            last_prompt_plan=last_prompt_plan,
            runtime_metrics=runtime_metrics or {},
        )
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return snapshot

