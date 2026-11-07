from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyChunk:
    program_id: str
    chunk_type: str
    text: str
    source_field: str


CHUNK_FIELDS = (
    ("PURPOSE", "purpose_summary"), ("PURPOSE_DETAIL", "purpose_detail"),
    ("ELIGIBILITY", "target"), ("SELECTION", "selection_criteria"),
    ("BENEFIT", "benefit"), ("APPLICATION", "application_method"),
    ("DOCUMENTS", "required_documents"),
)


def chunk_program(program: dict, max_chars: int = 600) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for chunk_type, field in CHUNK_FIELDS:
        value = str(program.get(field) or "").strip()
        if not value:
            continue
        for start in range(0, len(value), max_chars):
            chunks.append(PolicyChunk(str(program["id"]), chunk_type, value[start:start + max_chars], field))
    return chunks
