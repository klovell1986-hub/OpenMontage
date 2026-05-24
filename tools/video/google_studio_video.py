"""Google AI Studio — Veo 2 free-tier video generation.

Uses the official google-genai SDK to submit text-to-video jobs against
Google AI Studio's free quota (50 clips/day, 5s or 8s, up to 720p).

This is the zero-cost path. For production quality / higher throughput use
veo_video.py (fal.ai backend, paid per second).

Requirements:
    pip install google-genai
    GEMINI_API_KEY=<your AI Studio key>  (in .env or environment)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class GoogleStudioVideo(BaseTool):
    """Google Veo 2 via AI Studio free tier — no billing required."""

    name = "google_studio_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "google_studio"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["google-genai"]
    install_instructions = (
        "pip install google-genai\n"
        "Set GEMINI_API_KEY to your Google AI Studio key.\n"
        "  Get one free at https://aistudio.google.com/apikey"
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": False,
        "reference_to_video": False,
        "first_last_frame_to_video": False,
        "native_audio": False,
        "free_tier": True,
    }
    best_for = [
        "zero-cost video generation (50 clips/day free quota)",
        "quick prototyping without a billing account",
        "short 5-8 second clips",
    ]
    not_good_for = [
        "production volume (50/day limit)",
        "clips longer than 8 seconds",
        "image-to-video workflows",
    ]
    fallback_tools = ["veo_video", "kling_video", "wan_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1"],
                "default": "16:9",
            },
            "duration": {
                "type": "integer",
                "enum": [5, 8],
                "default": 5,
                "description": "Clip duration in seconds (5 or 8 only on free tier)",
            },
            "output_path": {
                "type": "string",
                "description": "Local path to save the .mp4 file",
            },
            "poll_interval": {
                "type": "integer",
                "default": 15,
                "description": "Seconds between status polls (Veo typically takes 2-4 min)",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=150, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["prompt", "aspect_ratio", "duration"]
    side_effects = ["writes video file to output_path", "calls Google AI Studio API"]
    user_visible_verification = [
        "Confirm the .mp4 file exists and has non-zero size",
        "Preview the clip for motion quality",
    ]

    def _get_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY")

    def get_status(self) -> ToolStatus:
        if not self._get_api_key():
            return ToolStatus.UNAVAILABLE
        try:
            from google import genai  # noqa: F401
            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Free tier — no cost
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        # Veo 2 on AI Studio typically takes 2-4 minutes
        return 180.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="GEMINI_API_KEY not set. " + self.install_instructions,
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return ToolResult(
                success=False,
                error="google-genai not installed. Run: pip install google-genai",
            )

        prompt = inputs["prompt"]
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        duration = int(inputs.get("duration", 5))
        poll_interval = int(inputs.get("poll_interval", 15))

        # Free tier only supports 5s and 8s
        if duration not in (5, 8):
            duration = 5

        # Determine output path
        output_path = Path(
            inputs.get("output_path")
            or f"workspace/temp_assets/google_studio_{int(time.time())}.mp4"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()

        try:
            client = genai.Client(api_key=api_key)

            # Submit the async generation job
            operation = client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    person_generation="dont_allow",
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                    duration_seconds=duration,
                ),
            )

            # Poll until complete — operation.done is a bool property
            while not operation.done:
                time.sleep(poll_interval)
                # Correct SDK call: operations.get() with the operation name string
                operation = client.operations.get(operation)

            # Extract the first generated video
            generated_videos = operation.result.generated_videos
            if not generated_videos:
                return ToolResult(
                    success=False,
                    error="Veo returned no generated videos in the result",
                )

            video = generated_videos[0].video

            # Download and save — save() writes bytes to disk
            client.files.download(file=video)
            video.save(str(output_path))

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Google Studio video generation failed: {e}",
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            return ToolResult(
                success=False,
                error=f"Output file missing or empty after generation: {output_path}",
            )

        return ToolResult(
            success=True,
            data={
                "provider": "google_studio",
                "model": "veo-2.0-generate-001",
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": duration,
                "output": str(output_path),
                "free_tier": True,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="veo-2.0-generate-001",
        )
