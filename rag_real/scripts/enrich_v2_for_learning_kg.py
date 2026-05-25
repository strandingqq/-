from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "v2"
CHUNKS_PATH = V2_DIR / "knowledge_chunks.jsonl"
ALL_QUESTIONS_PATH = V2_DIR / "all_questions.jsonl"
TOPICS_PATH = V2_DIR / "topics.jsonl"
EDGES_PATH = V2_DIR / "kg_edges.jsonl"

QUESTION_PATHS = [
    ALL_QUESTIONS_PATH,
    *sorted(
        p
        for p in V2_DIR.glob("*_questions.jsonl")
        if p.name != ALL_QUESTIONS_PATH.name
    ),
    V2_DIR / "review_questions.jsonl",
]

DIFFICULTY_SCORE = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
SCORE_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard", 4: "hard"}

NON_ASCII_TOPIC_SLUGS = {
    "\u62f7\u8d1d\u6784\u9020": "copy_constructor",
    "\u79fb\u52a8\u6784\u9020": "move_constructor",
    "\u5185\u5b58\u6cc4\u6f0f": "memory_leak",
    "\u9003\u9038\u5206\u6790": "escape_analysis",
    "\u5f31\u5f15\u7528": "weak_reference",
    "\u52a8\u6001\u7279\u6027": "dynamic_features",
}

FOUNDATIONAL_TERMS = {
    "syntax",
    "arrays",
    "strings",
    "variables",
    "functions",
    "classes",
    "structs",
    "operators",
    "pointers",
    "references",
    "initialization",
    "html",
    "css",
    "dom",
    "http",
    "tcp",
    "udp",
}

ADVANCED_TERMS = {
    "atomic",
    "compiler",
    "concurrency",
    "coroutine",
    "deadlock",
    "distributed",
    "garbage collection",
    "gmp",
    "jit",
    "knowledge distillation",
    "lock",
    "memory model",
    "metaprogramming",
    "model compression",
    "optimization",
    "profiling",
    "quantization",
    "runtime",
    "template",
}

MANUAL_PREREQUISITES = {
    "cpp": [
        ("C++ Syntax", "C++11"),
        ("C++ Syntax", "C++ Standards"),
        ("C++ Standards", "C++20"),
        ("C++ Standards", "C++23"),
        ("C++11", "Move Semantics"),
        ("C++11", "std::atomic"),
        ("Memory Model", "std::atomic"),
        ("std::atomic", "Deadlock Prevention"),
        ("std::lock", "Deadlock Prevention"),
        ("Pointers", "Smart Pointers"),
        ("Classes", "Constructors"),
        ("Classes", "Destructors"),
        ("Classes", "Inheritance"),
        ("Templates", "Template Metaprogramming"),
        ("Templates", "SFINAE"),
        ("Templates", "Concepts"),
        ("C++20", "Modules (C++20)"),
        ("C++20", "Coroutines (C++20)"),
        ("C++20", "Concepts (C++20)"),
    ],
    "go": [
        ("Goroutines", "Channels"),
        ("Channels", "Buffered Channels"),
        ("Channels", "Unbuffered Channels"),
        ("Channels", "Close Channel"),
        ("Channels", "Range over Channel"),
        ("Goroutines", "Mutex vs Channel"),
        ("Context", "context.Context"),
        ("Goroutines", "go atomic"),
    ],
    "frontend": [
        ("JavaScript", "DOM Manipulation"),
        ("DOM Manipulation", "Web APIs"),
        ("State Management", "Redux"),
        ("State Management", "Vue"),
        ("JavaScript", "Event Loop"),
    ],
    "python": [
        ("Functions", "Decorators"),
        ("Context Managers", "with Statement"),
        ("Multithreading", "threading.Lock"),
        ("Testing", "Coverage"),
        ("Serialization", "Protocol Buffers"),
    ],
    "python_backend": [
        ("HTTP/HTTPS", "FastAPI"),
        ("HTTP/HTTPS", "Django"),
        ("Unit Testing", "Coverage"),
        ("Database", "SQLAlchemy"),
        ("API Design", "RESTful API"),
    ],
    "embedded": [
        ("Model Compression", "Quantization"),
        ("Model Compression", "Pruning"),
        ("Model Compression", "Knowledge Distillation"),
        ("Edge AI", "Model Deployment"),
        ("ONNX", "Model Deployment"),
    ],
    "cs_fundamentals": [
        ("Processes", "Context Switching"),
        ("Locking", "Deadlock"),
        ("TCP", "HTTP/HTTPS"),
        ("DNS", "HTTP/HTTPS"),
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ": ")))
            f.write("\n")


def unique(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def slugify_topic(topic: str) -> str:
    topic = topic.strip()
    if topic in NON_ASCII_TOPIC_SLUGS:
        return NON_ASCII_TOPIC_SLUGS[topic]

    slug = topic.lower()
    slug = slug.replace("c++", "cxx")
    slug = slug.replace("c#", "csharp")
    slug = slug.replace(".net", "dotnet")
    slug = slug.replace("std::", "std_")
    slug = slug.replace("::", "_")
    slug = slug.replace("+", "plus")
    slug = slug.replace("#", "sharp")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")

    if not slug:
        digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]
        slug = f"topic_{digest}"
    return slug


def build_topic_ids(topic_pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for role, topic in topic_pairs:
        grouped[(role, slugify_topic(topic))].append(topic)

    topic_ids: dict[tuple[str, str], str] = {}
    for (role, slug), topics in grouped.items():
        for topic in sorted(topics):
            final_slug = slug
            if len(topics) > 1:
                digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:8]
                final_slug = f"{slug}_{digest}"
            topic_ids[(role, topic)] = f"{role}:{final_slug}"
    return topic_ids


def infer_difficulty(question_records: list[dict[str, Any]]) -> str:
    counts = Counter(q.get("difficulty") for q in question_records if q.get("difficulty"))
    if not counts:
        return "medium"
    return counts.most_common(1)[0][0]


def infer_level(topic: str, difficulty: str) -> int:
    topic_lower = topic.lower()
    base = round(DIFFICULTY_SCORE.get(difficulty, 2.0))
    level = max(1, min(3, int(base)))

    if any(term in topic_lower for term in FOUNDATIONAL_TERMS):
        level = min(level, 2)
    if any(term in topic_lower for term in ADVANCED_TERMS):
        level = max(level, 3)
    if "cxx20" in slugify_topic(topic) or "cxx23" in slugify_topic(topic):
        level = max(level, 3)
    return level


def learning_stage_for_chunk(chunk_type: str) -> str:
    if chunk_type == "principle":
        return "concept"
    if chunk_type == "practice":
        return "practice"
    return "concept"


def ordered_chunk_record(record: dict[str, Any]) -> dict[str, Any]:
    field_order = [
        "chunk_id",
        "role",
        "role_label",
        "topic",
        "topic_id",
        "canonical_topic",
        "chunk_type",
        "learning_stage",
        "content",
        "key_points",
        "related_topics",
        "tags",
        "source_file",
        "source_line",
        "quality_flags",
    ]
    return order_record(record, field_order)


def ordered_question_record(record: dict[str, Any]) -> dict[str, Any]:
    field_order = [
        "question_id",
        "role",
        "role_label",
        "topic",
        "topic_id",
        "question_type",
        "difficulty",
        "question",
        "expected_answer",
        "reference_points",
        "follow_up_angles",
        "common_mistakes",
        "rubric",
        "tags",
        "assesses_topic_ids",
        "gold_chunk_ids",
        "source_file",
        "source_line",
        "quality_flags",
        "review_status",
    ]
    return order_record(record, field_order)


def order_record(record: dict[str, Any], field_order: list[str]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for field in field_order:
        if field in record:
            ordered[field] = record[field]
    for field, value in record.items():
        if field not in ordered:
            ordered[field] = value
    return ordered


def edge_record(
    source_id: str,
    source_type: str,
    relation: str,
    target_id: str,
    target_type: str,
    weight: float,
    confidence: float,
    evidence: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "relation": relation,
        "target_id": target_id,
        "target_type": target_type,
        "weight": round(weight, 3),
        "confidence": round(confidence, 3),
        "evidence": evidence,
    }


def add_edge(edges: dict[tuple[str, str, str], dict[str, Any]], edge: dict[str, Any]) -> None:
    key = (edge["source_id"], edge["relation"], edge["target_id"])
    previous = edges.get(key)
    if previous is None:
        edges[key] = edge
        return

    if edge["confidence"] > previous["confidence"]:
        edges[key] = edge
        return

    previous["weight"] = max(previous["weight"], edge["weight"])
    previous["confidence"] = max(previous["confidence"], edge["confidence"])


def resolve_topic_id(
    role: str,
    topic_name: str,
    topic_ids: dict[tuple[str, str], str],
    slug_index: dict[tuple[str, str], str],
    global_slug_index: dict[str, list[str]],
) -> str | None:
    if (role, topic_name) in topic_ids:
        return topic_ids[(role, topic_name)]

    slug = slugify_topic(topic_name)
    if (role, slug) in slug_index:
        return slug_index[(role, slug)]

    global_matches = global_slug_index.get(slug, [])
    if len(global_matches) == 1:
        return global_matches[0]
    return None


def main() -> None:
    chunks = read_jsonl(CHUNKS_PATH)
    all_questions = read_jsonl(ALL_QUESTIONS_PATH)
    question_paths = [path for path in QUESTION_PATHS if path.exists()]
    review_questions = (
        read_jsonl(V2_DIR / "review_questions.jsonl")
        if (V2_DIR / "review_questions.jsonl").exists()
        else []
    )
    topic_question_records = [*all_questions, *review_questions]

    topic_pairs = {(c["role"], c["topic"]) for c in chunks}
    topic_pairs.update((q["role"], q["topic"]) for q in topic_question_records)
    topic_ids = build_topic_ids(topic_pairs)

    slug_index: dict[tuple[str, str], str] = {}
    global_slug_index: dict[str, list[str]] = defaultdict(list)
    topic_by_id: dict[str, tuple[str, str]] = {}
    for (role, topic), topic_id in topic_ids.items():
        slug = topic_id.split(":", 1)[1]
        slug_index[(role, slug)] = topic_id
        global_slug_index[slug].append(topic_id)
        topic_by_id[topic_id] = (role, topic)

    chunks_by_topic: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    chunk_ids_by_topic: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk in chunks:
        key = (chunk["role"], chunk["topic"])
        topic_id = topic_ids[key]
        chunk["topic_id"] = topic_id
        chunk["canonical_topic"] = chunk["topic"]
        chunk["learning_stage"] = learning_stage_for_chunk(chunk.get("chunk_type", ""))
        chunks_by_topic[key].append(chunk)

    for key, key_chunks in chunks_by_topic.items():
        ordered_chunks = sorted(
            key_chunks,
            key=lambda c: (0 if c.get("chunk_type") == "principle" else 1, c["chunk_id"]),
        )
        chunk_ids_by_topic[key] = [c["chunk_id"] for c in ordered_chunks]

    questions_by_topic: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in topic_question_records:
        key = (question["role"], question["topic"])
        questions_by_topic[key].append(question)

    topics: list[dict[str, Any]] = []
    topic_level: dict[str, int] = {}
    for key in sorted(topic_ids, key=lambda item: topic_ids[item]):
        role, topic = key
        topic_id = topic_ids[key]
        topic_chunks = chunks_by_topic.get(key, [])
        topic_questions = questions_by_topic.get(key, [])
        difficulty = infer_difficulty(topic_questions)
        level = infer_level(topic, difficulty)
        topic_level[topic_id] = level

        alias_candidates: list[str] = [topic]
        learning_objectives: list[str] = []
        for chunk in sorted(topic_chunks, key=lambda c: c.get("chunk_type") != "principle"):
            alias_candidates.extend(chunk.get("tags", []) or [])
            alias_candidates.extend(chunk.get("related_topics", []) or [])
            learning_objectives.extend(chunk.get("key_points", []) or [])
        for question in topic_questions[:5]:
            alias_candidates.extend(question.get("tags", []) or [])

        topics.append(
            {
                "topic_id": topic_id,
                "role": role,
                "name": topic,
                "aliases": unique(alias_candidates, limit=12),
                "level": level,
                "difficulty": SCORE_DIFFICULTY.get(level, difficulty),
                "prerequisite_topic_ids": [],
                "learning_objectives": unique(learning_objectives, limit=6),
            }
        )

    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    for chunk in chunks:
        relation = "EXPLAINS" if chunk.get("chunk_type") == "principle" else "PRACTICES"
        add_edge(
            edges,
            edge_record(
                source_id=chunk["chunk_id"],
                source_type="chunk",
                relation=relation,
                target_id=chunk["topic_id"],
                target_type="topic",
                weight=1.0,
                confidence=1.0,
                evidence="chunk_type",
            ),
        )

    for chunk in chunks:
        current_topic_id = chunk["topic_id"]
        for related_topic in chunk.get("related_topics", []) or []:
            related_topic_id = resolve_topic_id(
                chunk["role"], related_topic, topic_ids, slug_index, global_slug_index
            )
            if not related_topic_id or related_topic_id == current_topic_id:
                continue

            add_edge(
                edges,
                edge_record(
                    source_id=current_topic_id,
                    source_type="topic",
                    relation="RELATED_TO",
                    target_id=related_topic_id,
                    target_type="topic",
                    weight=0.6,
                    confidence=0.55,
                    evidence="related_topics",
                ),
            )

            current_level = topic_level[current_topic_id]
            related_level = topic_level[related_topic_id]
            if current_level == related_level:
                continue
            source_id, target_id = (
                (related_topic_id, current_topic_id)
                if related_level < current_level
                else (current_topic_id, related_topic_id)
            )
            add_edge(
                edges,
                edge_record(
                    source_id=source_id,
                    source_type="topic",
                    relation="PREREQUISITE_OF",
                    target_id=target_id,
                    target_type="topic",
                    weight=0.7,
                    confidence=0.6,
                    evidence="related_topic_level_heuristic",
                ),
            )

    for role, pairs in MANUAL_PREREQUISITES.items():
        for source_topic, target_topic in pairs:
            source_id = resolve_topic_id(
                role, source_topic, topic_ids, slug_index, global_slug_index
            )
            target_id = resolve_topic_id(
                role, target_topic, topic_ids, slug_index, global_slug_index
            )
            if not source_id or not target_id or source_id == target_id:
                continue
            add_edge(
                edges,
                edge_record(
                    source_id=source_id,
                    source_type="topic",
                    relation="PREREQUISITE_OF",
                    target_id=target_id,
                    target_type="topic",
                    weight=0.9,
                    confidence=0.85,
                    evidence="manual_seed",
                ),
            )

    for path in question_paths:
        questions = read_jsonl(path)
        for question in questions:
            key = (question["role"], question["topic"])
            topic_id = topic_ids.get(key)
            if not topic_id:
                continue

            question["topic_id"] = topic_id
            question["assesses_topic_ids"] = [topic_id]
            question["gold_chunk_ids"] = chunk_ids_by_topic.get(key, [])

            add_edge(
                edges,
                edge_record(
                    source_id=question["question_id"],
                    source_type="question",
                    relation="ASSESSES",
                    target_id=topic_id,
                    target_type="topic",
                    weight=1.0,
                    confidence=1.0,
                    evidence="topic_match",
                ),
            )
            for rank, chunk_id in enumerate(question["gold_chunk_ids"]):
                add_edge(
                    edges,
                    edge_record(
                        source_id=question["question_id"],
                        source_type="question",
                        relation="GROUNDED_BY",
                        target_id=chunk_id,
                        target_type="chunk",
                        weight=1.0 if rank == 0 else 0.85,
                        confidence=0.9,
                        evidence="topic_match",
                    ),
                )

        write_jsonl(path, [ordered_question_record(q) for q in questions])

    prerequisite_map: dict[str, list[str]] = defaultdict(list)
    for edge in edges.values():
        if edge["relation"] == "PREREQUISITE_OF":
            prerequisite_map[edge["target_id"]].append(edge["source_id"])

    for topic in topics:
        topic["prerequisite_topic_ids"] = sorted(set(prerequisite_map[topic["topic_id"]]))

    edge_records = sorted(
        edges.values(),
        key=lambda e: (
            e["relation"],
            e["source_type"],
            e["source_id"],
            e["target_type"],
            e["target_id"],
        ),
    )

    write_jsonl(CHUNKS_PATH, [ordered_chunk_record(c) for c in chunks])
    write_jsonl(TOPICS_PATH, topics)
    write_jsonl(EDGES_PATH, edge_records)

    validate(chunks, question_paths, topics, edge_records)
    print(
        "enriched",
        len(chunks),
        "chunks,",
        sum(len(read_jsonl(path)) for path in question_paths),
        "question records,",
        len(topics),
        "topics,",
        len(edge_records),
        "edges",
    )


def validate(
    chunks: list[dict[str, Any]],
    question_paths: list[Path],
    topics: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    topic_ids = {topic["topic_id"] for topic in topics}
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    question_ids: set[str] = set()

    missing_chunk_topic = [
        chunk.get("chunk_id")
        for chunk in chunks
        if not chunk.get("topic_id")
        or not chunk.get("canonical_topic")
        or not chunk.get("learning_stage")
    ]
    if missing_chunk_topic:
        raise ValueError(f"chunks missing KG fields: {missing_chunk_topic[:5]}")

    for path in question_paths:
        for question in read_jsonl(path):
            question_ids.add(question["question_id"])
            if not question.get("topic_id") or not question.get("assesses_topic_ids"):
                raise ValueError(f"question missing KG fields in {path.name}: {question}")

    known_ids = {
        "topic": topic_ids,
        "chunk": chunk_ids,
        "question": question_ids,
    }
    for edge in edges:
        if edge["source_id"] not in known_ids[edge["source_type"]]:
            raise ValueError(f"unknown edge source: {edge}")
        if edge["target_id"] not in known_ids[edge["target_type"]]:
            raise ValueError(f"unknown edge target: {edge}")


if __name__ == "__main__":
    main()
