"""
Chain-of-Note (CoN) utilities.

Mode-A implementation:
- Teacher generates ReadingNote i sequentially (note i sees notes 1..i-1).
- Student outputs ReadingNote 1..k + FinalAnswer in one pass.

Paper-aligned CoN cases:
- relevant_find_answer
- irrelevant_infer_answer
- irrelevant_answer_unknown
"""

from .schema import ConBlock
from .teacher import TeacherGenConfig, LLMClient, generate_gold_notes_for_example

__all__ = [
    "ConBlock",
    "TeacherGenConfig",
    "LLMClient",
    "generate_gold_notes_for_example",
]
