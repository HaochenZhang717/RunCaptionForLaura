#!/usr/bin/env python3
"""Evaluate model answers as Relevant/Irrelevant using Qwen3-VL as a judge."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


PROMPT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "multimodal": {
        "include_image": True,
        "system_prompt": (
            "You are an expert in content evaluation. Your task is to determine whether the given answer "
            "to the image and question pair is Relevant or Irrelevant. "
            "Follow these guidelines strictly:\n\n"
            "1. An answer should be classified as 'Relevant' if it corresponds to the question and image context.\n\n"
            "2. An answer should be classified as 'Irrelevant' if it is off-topic, unrelated to the question/image, or contains nonsensical phrases.\n\n"
            "3. Relevance is about correspondence and topical alignment, not factual truthfulness. "
            "Do not mark an answer Irrelevant only because it is factually incorrect or misleading if it still clearly corresponds "
            "to the asked question.\n\n"
            "4. Balance both topical correspondence and answer quality when deciding. "
            "Always provide reasoning for your decision, and ensure your response strictly starts "
            "with either 'Relevant:' or 'Irrelevant:'."
        ),
        "user_prompt_template": (
            "Question: {question}\n\n"
            "Answer: {answer}\n\n"
            "Please analyze the provided answer in terms of its relevance to the image and question. "
            "Do not mark an answer Irrelevant only because it is factually incorrect or misleading. Provide a short explanation for your decision."
        ),
    },
    "text_only": {
        "include_image": False,
        "system_prompt": (
            "You are an expert in content evaluation. Your task is to determine whether the given answer "
            "is Relevant or Irrelevant to the provided textual context. "
            "The context may include an image caption followed by a question, but you should judge only from the "
            "text you are given and should not assume access to the image itself.\n\n"
            "1. An answer is 'Relevant' if it clearly responds to the textual context.\n\n"
            "2. An answer is 'Irrelevant' if it is off-topic, unrelated to the context, "
            "or nonsensical.\n\n"
            "3. Relevance is about topical alignment, not factual correctness.\n\n"
            "4. Provide brief reasoning and make sure your response starts with either 'Relevant:' or 'Irrelevant:'."
        ),
        "user_prompt_template": (
            "Textual context: {question}\n\n"
            "Answer: {answer}\n\n"
            "Please analyze whether the answer is relevant to the textual context only. "
            "Do not mark an answer Irrelevant only because it is factually incorrect or misleading. Provide a short explanation for your decision."
        ),
    },
    "image_only": {
        "include_image": True,
        "system_prompt": (
            "You are an expert in content evaluation. Your task is to determine whether the given answer "
            "is Relevant or Irrelevant to the provided image. There is no accompanying question.\n\n"
            "1. An answer is 'Relevant' if it clearly describes, summarizes, or otherwise corresponds to visible "
            "content in the image.\n\n"
            "2. An answer is 'Irrelevant' if it is off-topic, generic in a way that does not match the image, "
            "poorly written, or nonsensical.\n\n"
            "3. Relevance is about whether the answer matches the image content, not whether it is perfectly "
            "complete or factually precise.\n\n"
            "4. Provide brief reasoning and make sure your response starts with either 'Relevant:' or 'Irrelevant:'."
        ),
        "user_prompt_template": (
            "Answer: {answer}\n\n"
            "Please analyze whether the answer is relevant to the provided image only. "
            "Do not mark an answer Irrelevant only because it is factually incorrect or misleading. Provide a short explanation for your decision."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge answer relevance using Qwen3-VL-Instruct on an outputs JSON."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to model outputs JSON (dict with 'results' or a raw list).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Where to write judge outputs. Defaults to <input>_relevance_eval_qwen3vl.json",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Judge model id.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens for judge response.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (default: greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature if --do_sample is set.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p if --do_sample is set.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Tensor device.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map for model loading.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of evaluated rows (0 = all).",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=20,
        help="Write partial outputs every N evaluated rows (0 disables).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards (for multi-GPU/process evaluation).",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to model and processor loaders.",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=("auto", "multimodal", "text_only", "image_only"),
        default="auto",
        help="Prompt preset for judging. 'auto' infers the variant from each row.",
    )
    return parser.parse_args()


def choose_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported input JSON format. Expected dict with 'results' or list.")


def load_judge_model(
    model_id: str,
    dtype: torch.dtype,
    device_map: str,
    trust_remote_code: bool,
):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
            return processor, model
        except Exception:
            pass

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    return processor, model


def infer_prompt_variant(question: str, image_path: Optional[Path]) -> str:
    has_question = bool(question.strip())
    has_image = image_path is not None
    if has_image and has_question:
        return "multimodal"
    if has_image:
        return "image_only"
    return "text_only"


def build_messages(
    prompt_variant: str,
    image_path: Optional[str],
    question: str,
    answer: str,
) -> List[Dict[str, Any]]:
    config = PROMPT_CONFIGS[prompt_variant]
    user_content: List[Dict[str, str]] = []
    if config["include_image"] and image_path is not None:
        user_content.append({"type": "image", "image": image_path})
    user_content.append(
        {
            "type": "text",
            "text": config["user_prompt_template"].format(question=question, answer=answer),
        }
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": config["system_prompt"]}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_template_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # apply_chat_template works with {"type":"image"} + external image tensor.
    out: List[Dict[str, Any]] = []
    for msg in messages:
        msg_copy = {"role": msg.get("role"), "content": []}
        for part in msg.get("content", []):
            part_type = part.get("type")
            if part_type == "image":
                msg_copy["content"].append({"type": "image"})
            elif part_type == "text":
                msg_copy["content"].append({"type": "text", "text": part.get("text", "")})
        out.append(msg_copy)
    return out


def parse_label(judge_text: str) -> Tuple[Optional[str], str]:
    text = (judge_text or "").strip()
    lowered = text.lower()
    if lowered.startswith("relevant:"):
        return "Relevant", text[len("Relevant:") :].strip()
    if lowered.startswith("irrelevant:"):
        return "Irrelevant", text[len("Irrelevant:") :].strip()
    return None, text


def resolve_image_path_for_eval(image_value: Any, input_json: Path) -> Optional[Path]:
    if not isinstance(image_value, str):
        return None
    raw = image_value.strip()
    if not raw or raw == ".":
        return None

    path = Path(raw)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((Path.cwd() / path).resolve())
        candidates.append((input_json.parent / path).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_payload(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    meta_input: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    relevant_count = sum(1 for x in results if x.get("classification") == "Relevant")
    irrelevant_count = sum(1 for x in results if x.get("classification") == "Irrelevant")
    unparseable_count = sum(1 for x in results if x.get("classification") is None and x.get("error") is None)
    error_count = sum(1 for x in results if x.get("error") is not None)
    skipped_count = sum(1 for x in results if x.get("skipped", False))

    summary = {
        "input_rows": len(rows),
        "shard_rows": sum(
            1 for idx in range(len(rows)) if idx % args.num_shards == args.shard_index
        ),
        "evaluated_rows": len(results),
        "relevant_count": relevant_count,
        "irrelevant_count": irrelevant_count,
        "unparseable_count": unparseable_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
    }

    return {
        "meta": {
            "input_json": str(args.input_json),
            "output_json": str(args.output_json),
            "judge_model_id": args.model_id,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": args.device_map,
            "limit": args.limit,
            "flush_every": args.flush_every,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "prompt_variant": args.prompt_variant,
            "input_meta": meta_input,
        },
        "summary": summary,
        "results": results,
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.flush_every < 0:
        raise ValueError("--flush_every must be >= 0")

    output_json = args.output_json or args.input_json.with_name(
        f"{args.input_json.stem}_relevance_eval_qwen3vl.json"
    )
    args.output_json = output_json

    with args.input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = load_rows(payload)
    meta_input = payload.get("meta") if isinstance(payload, dict) else None

    dtype = choose_dtype(args.dtype)
    device = choose_device(args.device)
    processor, model = load_judge_model(
        model_id=args.model_id,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    results: List[Dict[str, Any]] = []
    progress = tqdm(rows, desc="Qwen3-VL relevance eval")

    for idx, row in enumerate(progress):
        if idx % args.num_shards != args.shard_index:
            continue
        if args.limit > 0 and len(results) >= args.limit:
            break

        question = (row.get("instruction") or "").strip()
        answer = (row.get("model_response") or "").strip()
        image_path = row.get("image")
        resolved_image_path = resolve_image_path_for_eval(image_path, args.input_json)
        upstream_error = row.get("error")
        breakpoint()
        output_item: Dict[str, Any] = {
            "row_index": idx,
            "id": row.get("id"),
            "source_index": row.get("source_index"),
            "turn_index": row.get("turn_index"),
            "global_prompt_index": row.get("global_prompt_index"),
            "prompt_type": row.get("prompt_type"),
            "safe": row.get("safe"),
            "image": image_path,
            "image_used_for_judge": str(resolved_image_path) if resolved_image_path else None,
            "question": question,
            "answer": answer,
            "prompt_variant_used": None,
            "upstream_error": upstream_error,
            "classification": None,
            "judge_response": None,
            "reasoning": None,
            "skipped": False,
            "error": None,
        }

        prompt_variant = args.prompt_variant
        if prompt_variant == "auto":
            prompt_variant = infer_prompt_variant(question=question, image_path=resolved_image_path)
        output_item["prompt_variant_used"] = prompt_variant

        if upstream_error is not None:
            output_item["skipped"] = True
            output_item["error"] = f"Skipped due to upstream generation error: {upstream_error}"
            results.append(output_item)
            continue
        if not answer:
            output_item["skipped"] = True
            output_item["error"] = "Missing answer text."
            results.append(output_item)
            continue

        try:
            image = None
            if resolved_image_path is not None:
                with Image.open(resolved_image_path) as img:
                    image = img.convert("RGB")

            messages = build_messages(
                prompt_variant=prompt_variant,
                image_path=str(resolved_image_path) if resolved_image_path else None,
                question=question,
                answer=answer,
            )
            template_messages = build_template_messages(messages)
            prompt = processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if image is None:
                model_inputs = processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                )
            else:
                model_inputs = processor(
                    text=[prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                output_ids = model.generate(**model_inputs, **generation_kwargs)

            generated_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]
            judge_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            label, reasoning = parse_label(judge_text)
            output_item["judge_response"] = judge_text
            output_item["classification"] = label
            output_item["reasoning"] = reasoning
        except Exception as exc:
            output_item["error"] = str(exc)

        results.append(output_item)

        if args.flush_every > 0 and len(results) % args.flush_every == 0:
            partial_payload = build_payload(args=args, rows=rows, results=results, meta_input=meta_input)
            write_json_atomic(args.output_json, partial_payload)
            progress.write(
                f"[shard {args.shard_index}] Flushed {len(results)} rows to {args.output_json}"
            )

    return build_payload(args=args, rows=rows, results=results, meta_input=meta_input)


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    write_json_atomic(args.output_json, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
