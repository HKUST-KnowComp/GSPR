---
name: gspr3-format-check
description: Validate incoming GSPR3 JSONL benchmark and training files for schema, message structure, labels, categories, image alignment, SFT output format, and conflicting annotations. Use when inspecting, accepting, repairing, or preparing new GSPR3 JSONL data for Slime RL/SFT training.
---

# GSPR3 format check

Use the bundled checker before merging or training on any incoming JSONL file. Treat validation errors as blockers until the source data is repaired or explicitly excluded. Preserve the original file and write reports or cleaned output separately.

## Checkpoints

Run the following checks in order and report each result separately.

1. **JSONL framing**
   - Require one JSON object per non-empty line.
   - Record invalid JSON lines, blank lines, and total row count.

2. **Top-level schema**
   - Require either `messages` (SFT/conversation format) or `prompt` (RL prompt format).
   - Require `images` as a list; an empty list is valid only for text-only rows.
   - Require `extra_info` as an object with a non-empty string `category` and a non-empty `category_dict` mapping (or JSON-encoded mapping) that covers the row's category.
   - Require `ground_truth` for RL rows when labels are external; if present in SFT rows, validate it too.

3. **Message structure for SFT**
   - Require a non-empty message list.
   - Require each message to be an object with role in `system`, `user`, `assistant`, or `tool`.
   - Require content to be a string, object, or list of content segments.
   - Require at least one assistant message for SFT data.

4. **Image alignment**
   - Count `<image>` placeholders across message text and compare them with `images`.
   - Reject image placeholders with an empty image list.
   - Accept image paths, URLs, data URIs, or rich image dictionaries.
   - For rich dictionaries, validate `type`, `image`, `min_pixels`, `max_pixels`, and `max_pixels >= min_pixels`.
   - Use `--check-image-paths` when local image existence should be checked; URLs and data URIs are exempt.

5. **Safety labels**
   - Require `ground_truth` to be exactly lowercase `safe` or `unsafe`; do not silently normalize other values.
   - Build a safe/unsafe distribution and identify missing or invalid labels.

6. **SFT target format**
   - Require exactly one assistant `\\safety{safe|unsafe}` label.
   - Require exactly one assistant `\\category{...}` label.
   - Compare `\\safety{...}` with `ground_truth` when both exist.
   - Compare `\\category{...}` with `extra_info.category`; category disagreement is a warning by default and should be reviewed.

7. **Taxonomy consistency**
   - Summarize `extra_info.category` values and inspect unexpected spelling, casing, whitespace, or benchmark-specific aliases.
   - Do not merge categories merely because they look similar; map aliases explicitly in a separate normalization step.

8. **Exact-prompt conflicts**
   - Hash the user/prompt portion while excluding assistant answers.
   - Report any exact conversation/prompt group containing both `safe` and `unsafe` labels.
   - Remove or adjudicate these groups before training; never select one label arbitrarily.

9. **Training suitability review**
   - Check that text-only rows have `images: []` and no `<image>` marker.
   - Check that multimodal rows have matching placeholders and image data.
   - For SFT, confirm every assistant answer contains only the intended safety/category target format (reasoning is allowed only if the training policy permits it).
   - Run a separate tokenizer/processor length audit for multimodal data before training; this schema checker does not estimate model token length.

## Run the checker

The checker is streaming and does not load the complete file into memory:

```bash
python /data3/haoran/GSPR3/skill/gspr3-format-check/scripts/check_gspr3_jsonl.py \
  /path/to/incoming.jsonl \
  --mode auto \
  --report /path/to/incoming.format_report.json
```

Use an explicit mode when the file type is known:

```bash
python /data3/haoran/GSPR3/skill/gspr3-format-check/scripts/check_gspr3_jsonl.py data.jsonl --mode sft --report data.report.json
python /data3/haoran/GSPR3/skill/gspr3-format-check/scripts/check_gspr3_jsonl.py data.jsonl --mode rl  --report data.report.json
```

Add `--check-image-paths` for datasets whose `images` entries are local paths. The command exits 0 when no errors are found, 1 when validation errors are present, and 2 when the input cannot be read.

## Handling findings

- Keep the JSON report as an audit artifact.
- Fix source rows or create a new filtered file; do not overwrite the incoming file during inspection.
- Treat `invalid_json`, schema violations, invalid binary labels, image mismatches, missing SFT targets, and exact-prompt conflicts as errors.
- Review category mismatches, missing local image paths, and blank lines as warnings or errors according to the dataset contract.
- After repair, rerun the checker and compare row counts and distributions with the previous report.

## Bundled resource

- `scripts/check_gspr3_jsonl.py` — deterministic streaming validator and JSON report generator.
