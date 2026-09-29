#!/usr/bin/env python3
"""Validate incoming GSPR3 JSONL files.

The checker is streaming and dependency-free. It validates schema, message
roles/content, image placeholder alignment, binary labels, SFT output labels,
metadata categories, and exact-prompt label conflicts. It writes a JSON report
and returns exit status 1 when any error is found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROLES = {"system", "user", "assistant", "tool"}
LABELS = {"safe", "unsafe"}
SAFETY_RE = re.compile(r"\\safety\{(safe|unsafe)\}")
CATEGORY_RE = re.compile(r"\\category\{([^{}\n]+)\}")


class Checker:
    def __init__(self, mode: str, max_findings: int, check_image_paths: bool):
        self.mode = mode
        self.max_findings = max_findings
        self.check_image_paths = check_image_paths
        self.findings: list[dict[str, Any]] = []
        self.counts = Counter()
        self.categories = Counter()
        self.benchmark_categories: dict[str, Counter] = defaultdict(Counter)
        self.benchmark_taxonomy: dict[str, set[str]] = defaultdict(set)
        self.labels = Counter()
        self.prompt_labels: dict[str, set[str]] = defaultdict(set)
        self.prompt_rows: dict[str, list[int]] = defaultdict(list)

    def finding(self, level: str, code: str, row: int | None, message: str) -> None:
        self.counts[level] += 1
        if len(self.findings) < self.max_findings:
            self.findings.append({"level": level, "code": code, "row": row, "message": message})

    def error(self, code: str, row: int | None, message: str) -> None:
        self.finding("error", code, row, message)

    def warning(self, code: str, row: int | None, message: str) -> None:
        self.finding("warning", code, row, message)

    @staticmethod
    def text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif item.get("type") == "image":
                        parts.append("<image>")
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return ""

    @classmethod
    def message_text(cls, messages: list[Any], include_assistant: bool = True) -> str:
        selected = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if not include_assistant and message.get("role") == "assistant":
                continue
            selected.append({"role": message.get("role"), "content": cls.text_from_content(message.get("content"))})
        encoded = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def check_images(self, row: dict[str, Any], messages: list[Any], row_no: int) -> None:
        if "images" not in row:
            self.error("missing_images", row_no, "missing required 'images' list")
            images: list[Any] = []
        else:
            images = row["images"]
            if not isinstance(images, list):
                self.error("images_not_list", row_no, "'images' must be a list")
                images = []

        placeholder_count = sum(self.text_from_content(m.get("content") if isinstance(m, dict) else "").count("<image>") for m in messages)
        embedded_count = 0
        for image in images:
            if isinstance(image, str):
                value = image
            elif isinstance(image, dict):
                if image.get("type") not in (None, "image"):
                    self.error("invalid_image_type", row_no, "image dict 'type' must be 'image'")
                value = image.get("image")
                if value is None:
                    self.error("missing_image_value", row_no, "image dict must contain 'image'")
                min_pixels, max_pixels = image.get("min_pixels"), image.get("max_pixels")
                if min_pixels is not None and (not isinstance(min_pixels, (int, float)) or min_pixels <= 0):
                    self.error("invalid_min_pixels", row_no, "min_pixels must be a positive number")
                if max_pixels is not None and (not isinstance(max_pixels, (int, float)) or max_pixels <= 0):
                    self.error("invalid_max_pixels", row_no, "max_pixels must be a positive number")
                if isinstance(min_pixels, (int, float)) and isinstance(max_pixels, (int, float)) and max_pixels < min_pixels:
                    self.error("pixel_range", row_no, "max_pixels must be >= min_pixels")
            else:
                self.error("invalid_image_item", row_no, "each image must be a path/URL/data URI or image dict")
                value = None
            if isinstance(value, str) and self.check_image_paths and not (value.startswith(("http://", "https://", "data:")) or Path(value).exists()):
                self.warning("missing_image_path", row_no, f"image path does not exist: {value[:160]}")

        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                embedded_count += sum(1 for item in message["content"] if isinstance(item, dict) and item.get("type") == "image")
        if images and placeholder_count != len(images) and embedded_count != len(images):
            self.error("image_placeholder_mismatch", row_no, f"{len(images)} images but {placeholder_count} <image> placeholders")
        if not images and placeholder_count:
            self.error("image_without_data", row_no, "prompt contains <image> but images is empty")

    def check_messages(self, messages: Any, row_no: int) -> tuple[list[Any], str | None, str | None]:
        if not isinstance(messages, list) or not messages:
            self.error("invalid_messages", row_no, "'messages' must be a non-empty list")
            return [], None, None
        assistant_texts = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                self.error("message_not_object", row_no, f"messages[{index}] must be an object")
                continue
            role = message.get("role")
            if role not in ROLES:
                self.error("invalid_role", row_no, f"messages[{index}].role={role!r} is invalid")
            if "content" not in message or not isinstance(message["content"], (str, list, dict)):
                self.error("invalid_content", row_no, f"messages[{index}].content must be string, list, or object")
            if role == "assistant":
                assistant_texts.append(self.text_from_content(message.get("content")))
        if self.mode == "sft" and not assistant_texts:
            self.error("missing_assistant", row_no, "SFT row has no assistant message")
        all_assistant = "\n".join(assistant_texts)
        return messages, all_assistant, self.message_text(messages, include_assistant=False)

    def check_row(self, row: Any, row_no: int) -> None:
        if not isinstance(row, dict):
            self.error("row_not_object", row_no, "each JSONL row must be an object")
            return

        row_mode = self.mode
        if row_mode == "auto":
            row_mode = "sft" if "messages" in row else "rl" if "prompt" in row else "unknown"
        if row_mode == "unknown":
            self.error("unknown_schema", row_no, "expected 'messages' (SFT) or 'prompt' (RL)")
            return

        messages: list[Any] = []
        assistant_text = None
        prompt_hash = None
        if "messages" in row:
            messages, assistant_text, prompt_hash = self.check_messages(row["messages"], row_no)
            self.check_images(row, messages, row_no)
        elif "prompt" in row:
            prompt = row["prompt"]
            if not isinstance(prompt, (str, list)):
                self.error("invalid_prompt", row_no, "'prompt' must be a string or message list")
            if isinstance(prompt, str):
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                self.check_images(row, [{"role": "user", "content": prompt}], row_no)
            elif isinstance(prompt, list):
                messages, assistant_text, prompt_hash = self.check_messages(prompt, row_no)
                self.check_images(row, messages, row_no)
        else:
            self.error("missing_prompt", row_no, "missing 'messages' or 'prompt'")

        if "images" not in row and "messages" not in row and "prompt" not in row:
            self.error("missing_images", row_no, "missing required 'images' list")

        label = row.get("ground_truth")
        valid_label = isinstance(label, str) and label in LABELS
        if label is not None:
            if not valid_label:
                self.error("invalid_ground_truth", row_no, f"ground_truth must be exactly 'safe' or 'unsafe', got {label!r}")
            else:
                self.labels[label] += 1
        elif row_mode == "rl":
            self.error("missing_ground_truth", row_no, "RL row requires ground_truth")

        metadata = row.get("extra_info")
        if not isinstance(metadata, dict):
            self.error("invalid_extra_info", row_no, "extra_info must be an object")
        else:
            category = metadata.get("category")
            if not isinstance(category, str) or not category.strip():
                self.error("missing_category", row_no, "extra_info.category must be a non-empty string")
            else:
                self.categories[category] += 1
                if "category_dict" not in metadata:
                    self.error("missing_category_dict", row_no, "extra_info.category_dict is required and must list the categories covered by the prompt")
                else:
                    category_dict = metadata["category_dict"]
                    if isinstance(category_dict, str):
                        if not category_dict.strip():
                            self.error("empty_category_dict", row_no, "extra_info.category_dict must not be empty")
                        else:
                            try:
                                category_dict = json.loads(category_dict)
                            except json.JSONDecodeError as exc:
                                self.error("invalid_category_dict", row_no, f"extra_info.category_dict must be a JSON object: {exc.msg}")
                                category_dict = None
                    if category_dict is not None and not isinstance(category_dict, dict):
                        self.error("invalid_category_dict", row_no, "extra_info.category_dict must be a mapping or a JSON-encoded mapping")
                    elif isinstance(category_dict, dict):
                        if not category_dict:
                            self.error("empty_category_dict", row_no, "extra_info.category_dict must contain at least one category")
                        keys = {str(key).strip() for key in category_dict}
                        values = {str(value).strip() for value in category_dict.values()}
                        # Safe examples conventionally use ``not applicable``;
                        # that sentinel is valid even when category_dict lists
                        # only the benchmark's unsafe categories.
                        if category != "not applicable" and category not in keys and category not in values:
                            self.error("category_not_in_dict", row_no, f"extra_info.category.category={category!r} is not covered by category_dict")
                benchmark = str(metadata.get("benchmark") or metadata.get("dataset") or "<unknown>")
                self.benchmark_categories[benchmark][category] += 1
                taxonomy = metadata.get("category_dict")
                if taxonomy is not None:
                    if isinstance(taxonomy, str):
                        try:
                            taxonomy = json.loads(taxonomy)
                        except json.JSONDecodeError:
                            pass
                    self.benchmark_taxonomy[benchmark].add(
                        json.dumps(taxonomy, ensure_ascii=False, sort_keys=True, default=str)
                    )

        if row_mode == "sft" and assistant_text is not None:
            safety = SAFETY_RE.findall(assistant_text)
            categories = CATEGORY_RE.findall(assistant_text)
            if len(safety) != 1:
                self.error("sft_safety_label", row_no, "SFT assistant response must contain exactly one \\safety{safe|unsafe}")
            if len(categories) != 1:
                self.error("sft_category_label", row_no, "SFT assistant response must contain exactly one \\category{...}")
            if safety and valid_label and safety[0] != label:
                self.error("label_mismatch", row_no, f"assistant safety label {safety[0]!r} disagrees with ground_truth {label!r}")
            if categories and isinstance(metadata, dict) and isinstance(metadata.get("category"), str) and categories[0] != metadata["category"]:
                self.warning("category_mismatch", row_no, f"assistant category {categories[0]!r} differs from extra_info.category {metadata['category']!r}")
            if prompt_hash and not valid_label and len(safety) == 1:
                self.prompt_labels[prompt_hash].add(safety[0])
                if len(self.prompt_rows[prompt_hash]) < 20:
                    self.prompt_rows[prompt_hash].append(row_no)

        if prompt_hash and valid_label:
            self.prompt_labels[prompt_hash].add(label)
            if len(self.prompt_rows[prompt_hash]) < 20:
                self.prompt_rows[prompt_hash].append(row_no)

    def finish(self) -> None:
        for digest, labels in self.prompt_labels.items():
            if labels == LABELS:
                rows = self.prompt_rows[digest]
                self.error("exact_prompt_label_conflict", rows[0] if rows else None, f"same prompt has both safe and unsafe labels (example rows: {rows})")
        for benchmark, taxonomies in self.benchmark_taxonomy.items():
            if len(taxonomies) > 1:
                if benchmark == "<unknown>":
                    self.warning("unknown_benchmark_taxonomy", None, "rows without benchmark metadata use multiple category taxonomies; add extra_info.benchmark to verify them separately")
                else:
                    self.error("benchmark_taxonomy_conflict", None, f"benchmark {benchmark!r} contains multiple category taxonomies")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("auto", "sft", "rl"), default="auto")
    parser.add_argument("--report", type=Path, help="write detailed JSON report")
    parser.add_argument("--max-findings", type=int, default=1000)
    parser.add_argument("--check-image-paths", action="store_true")
    args = parser.parse_args()

    checker = Checker(args.mode, args.max_findings, args.check_image_paths)
    rows = 0
    blank = 0
    try:
        with args.input.open("r", encoding="utf-8") as stream:
            for row_no, line in enumerate(stream, 1):
                if not line.strip():
                    blank += 1
                    continue
                rows += 1
                try:
                    checker.check_row(json.loads(line), row_no)
                except json.JSONDecodeError as exc:
                    checker.error("invalid_json", row_no, f"invalid JSON: {exc.msg}")
                except Exception as exc:  # keep checking the remaining file
                    checker.error("checker_exception", row_no, repr(exc))
    except OSError as exc:
        print(f"ERROR: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2

    checker.finish()
    report = {
        "input": str(args.input),
        "mode": args.mode,
        "rows": rows,
        "blank_lines": blank,
        "errors": checker.counts["error"],
        "warnings": checker.counts["warning"],
        "ground_truth_distribution": dict(checker.labels),
        "category_distribution": dict(checker.categories),
        "benchmark_category_distribution": {k: dict(v) for k, v in checker.benchmark_categories.items()},
        "benchmark_taxonomy_variants": {k: len(v) for k, v in checker.benchmark_taxonomy.items()},
        "findings_truncated": max(0, checker.counts["error"] + checker.counts["warning"] - len(checker.findings)),
        "findings": checker.findings,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("input", "mode", "rows", "blank_lines", "errors", "warnings", "ground_truth_distribution", "findings_truncated")}, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
