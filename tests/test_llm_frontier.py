from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.curriculum import generate_teacher_piece
from pixie_solver.llm import (
    FrontierLLMClient,
    FrontierLLMPieceProgramProvider,
    LLMConfig,
)
from pixie_solver.rules import CompileRequest, RepairRequest
from pixie_solver.rules.repair import default_dsl_reference
from pixie_solver.utils import canonical_json


class FrontierLLMTest(unittest.TestCase):
    def test_anthropic_compile_uses_claude_46_adaptive_thinking(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        transport = _RecordingTransport(
            {
                "id": "msg_test",
                "content": [
                    {"type": "thinking", "thinking": ""},
                    {
                        "type": "text",
                        "text": canonical_json(
                            {
                                "candidate_program": teacher.teacher_program,
                                "explanation": "compiled",
                            }
                        ),
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )
        provider = FrontierLLMPieceProgramProvider(
            FrontierLLMClient(
                LLMConfig(provider="anthropic", api_key="test-anthropic-key"),
                transport,
            )
        )

        response = provider.compile_piece(
            CompileRequest(
                description=teacher.description,
                dsl_reference=default_dsl_reference(),
            )
        )

        call = transport.calls[0]
        self.assertEqual("https://api.anthropic.com/v1/messages", call["url"])
        self.assertEqual("test-anthropic-key", call["headers"]["x-api-key"])
        self.assertEqual("claude-opus-4-6", call["payload"]["model"])
        self.assertEqual({"type": "adaptive", "display": "omitted"}, call["payload"]["thinking"])
        self.assertEqual({"effort": "high"}, call["payload"]["output_config"])
        self.assertEqual(teacher.teacher_program["piece_id"], response.candidate_program["piece_id"])
        self.assertEqual("anthropic", response.metadata["provider"])
        self.assertNotIn("test-anthropic-key", canonical_json(response.metadata))

    def test_openai_repair_uses_responses_api_and_reasoning_effort(self) -> None:
        teacher = generate_teacher_piece(seed=3, recipe="phase_rook")
        transport = _RecordingTransport(
            {
                "id": "resp_test",
                "output_text": "```json\n"
                + canonical_json(
                    {
                        "patched_program": teacher.teacher_program,
                        "explanation": "restored phasing",
                    }
                )
                + "\n```",
                "usage": {"input_tokens": 11, "output_tokens": 22},
            }
        )
        provider = FrontierLLMPieceProgramProvider(
            FrontierLLMClient(
                LLMConfig(
                    provider="openai",
                    api_key="test-openai-key",
                    effort="xhigh",
                ),
                transport,
            )
        )

        response = provider.repair_piece(
            RepairRequest(
                description=teacher.description,
                current_program=teacher.teacher_program,
                before_state={},
                move={},
                predicted_state={},
                observed_state={},
                diff={},
                predicted_error="illegal move: ally blocker",
                dsl_reference=default_dsl_reference(),
            )
        )

        call = transport.calls[0]
        self.assertEqual("https://api.openai.com/v1/responses", call["url"])
        self.assertEqual("Bearer test-openai-key", call["headers"]["authorization"])
        self.assertEqual("gpt-5.4", call["payload"]["model"])
        self.assertEqual({"effort": "xhigh"}, call["payload"]["reasoning"])
        self.assertIn("predicted_error", call["payload"]["input"][1]["content"][0]["text"])
        self.assertEqual(teacher.teacher_program["piece_id"], response.patched_program["piece_id"])
        self.assertEqual("openai", response.metadata["provider"])
        self.assertNotIn("test-openai-key", canonical_json(response.metadata))


class _RecordingTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


if __name__ == "__main__":
    unittest.main()
