"""
Rewrite each safe_instruction so that its coarse sentence form matches the
paired unsafe_instruction, while preserving the original safe meaning.

Only two sentence forms are used:
    - question
    - instruction

This script is intended for files with the test.json-style structure:
    item["instr-resp"] = [
        {"safe_instruction": "...", "response": "..."},
        {"unsafe_instruction": "...", "response": "..."},
    ]

The paired unsafe_instruction is used only to decide which form the safe
instruction should take. unsafe_instruction itself is never modified.

Output files are saved next to the input files with suffix
    _safe_style_matched.json
unless --out_suffix is specified.
"""

import argparse
import copy
import json
import os
import re
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILES = ["test.json"]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B"

FORM_EXAMPLES = """Examples:
TARGET FORM: question
SAFE PROMPT: Describe the main product shown in the image.
OUTPUT: What is the main product shown in the image?

TARGET FORM: instruction
SAFE PROMPT: What breed of dog might be depicted in the image?
OUTPUT: Describe the breed of dog that might be depicted in the image.

TARGET FORM: instruction
SAFE PROMPT: What is the main purpose of the tool advertised in the image?
OUTPUT: Describe the main purpose of the tool advertised in the image.
"""

BASE_PROMPT_TEMPLATE = (
    "Task: Rewrite SAFE PROMPT so it keeps its original safe meaning while "
    "matching TARGET FORM.\n"
    "Rules:\n"
    "- Preserve the meaning of SAFE PROMPT.\n"
    "- TARGET FORM must be either question or instruction.\n"
    "- If TARGET FORM is question, output one natural question ending with '?'.\n"
    "- If TARGET FORM is instruction, output one imperative instruction and do not end with '?'.\n"
    "- Do not add refusals, warnings, or policy language.\n"
    "- Output exactly one rewritten prompt and nothing else.\n\n"
    f"{FORM_EXAMPLES}\n"
    "TARGET FORM: {target_form}\n"
    "SAFE PROMPT: {safe_prompt}\n"
    "OUTPUT:"
)

SYSTEM_PROMPT = (
    "You rewrite a safe image-related prompt so that it matches a target "
    "sentence form: either question or instruction. Preserve the safe prompt's "
    "meaning. Output only the rewritten prompt."
)

CHAT_USER_TEMPLATE = (
    f"{FORM_EXAMPLES}\n"
    "TARGET FORM: {target_form}\n"
    "SAFE PROMPT: {safe_prompt}\n"
    "OUTPUT:"
)

REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi can not\b",
    r"\bi won'?t\b",
    r"\bi will not\b",
    r"\bsorry\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bcannot help\b",
    r"\bcan'?t help\b",
    r"\billegal\b",
    r"\bharmful\b",
    r"\bunethical\b",
    r"\bagainst (the )?law\b",
    r"\bnot able to\b",
    r"\bnot provide\b",
]

STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "could", "describe", "do", "for",
    "from", "give", "how", "identify", "image", "in", "is", "it", "kind",
    "list", "me", "might", "of", "on", "personal", "provide", "shown", "tell",
    "that", "the", "these", "this", "to", "using", "what", "which", "with",
    "would", "you", "your",
}

def clean_output(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "“", "”"):
        text = text[1:-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def has_refusal_language(text: str) -> bool:
    lowered = text.lower().strip()
    return any(re.search(pattern, lowered) for pattern in REFUSAL_PATTERNS)


def detect_form(text: str) -> str:
    return "question" if text.strip().endswith("?") else "instruction"


def normalize_to_form(text: str, target_form: str) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return text
    if target_form == "question":
        text = text.rstrip(" .!")
        if not text.endswith("?"):
            text += "?"
    else:
        text = text.rstrip()
        if text.endswith("?"):
            text = text[:-1].rstrip()
    return text


def content_tokens(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", text.lower())
    return {tok for tok in tokens if len(tok) >= 4 and tok not in STOPWORDS}


def is_valid_rewrite(output: str, safe_prompt: str, target_form: str) -> bool:
    if not output or has_refusal_language(output):
        return False

    out_tokens = content_tokens(output)
    safe_tokens = content_tokens(safe_prompt)
    if safe_tokens and not (out_tokens & safe_tokens):
        return False

    if target_form == "question" and not output.endswith("?"):
        return False
    if target_form == "instruction" and output.endswith("?"):
        return False

    return True


def find_safe_unsafe_indices(pairs: List[dict]) -> Tuple[Optional[int], Optional[int]]:
    safe_idx = None
    unsafe_idx = None
    for i, pair in enumerate(pairs):
        if safe_idx is None and isinstance(pair.get("safe_instruction"), str):
            safe_idx = i
        if unsafe_idx is None and isinstance(pair.get("unsafe_instruction"), str):
            unsafe_idx = i
    return safe_idx, unsafe_idx


class StyleMatcher:
    def __init__(self, model_path: str, device: str = None, dtype=torch.bfloat16):
        self.model_path = model_path
        model_name = model_path.lower()
        self.use_chat_template = any(key in model_name for key in ("instruct", "chat"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model from {model_path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _build_prompt(self, safe_prompt: str, target_form: str) -> str:
        if not self.use_chat_template:
            return BASE_PROMPT_TEMPLATE.format(
                safe_prompt=safe_prompt,
                target_form=target_form,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CHAT_USER_TEMPLATE.format(
                    safe_prompt=safe_prompt,
                    target_form=target_form,
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def rewrite_batch(
        self,
        safe_prompts: List[str],
        target_forms: List[str],
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        prompts = [
            self._build_prompt(safe_prompt, target_form)
            for safe_prompt, target_form in zip(safe_prompts, target_forms)
        ]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        trimmed = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

        results = []
        for safe_prompt, target_form, raw in zip(safe_prompts, target_forms, decoded):
            cleaned = normalize_to_form(clean_output(raw), target_form)
            if is_valid_rewrite(cleaned, safe_prompt, target_form):
                results.append(cleaned)
            else:
                results.append(safe_prompt)
        return results


def process_file(
    in_path: str,
    out_path: str,
    matcher: StyleMatcher,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = copy.deepcopy(data)
    flat = []

    for item_idx, item in enumerate(out_data):
        pairs = item.get("instr-resp")
        if not isinstance(pairs, list):
            continue
        safe_idx, unsafe_idx = find_safe_unsafe_indices(pairs)
        if safe_idx is None or unsafe_idx is None:
            continue

        safe_prompt = pairs[safe_idx]["safe_instruction"]
        unsafe_prompt = pairs[unsafe_idx]["unsafe_instruction"]
        target_form = detect_form(unsafe_prompt)
        flat.append((item_idx, safe_idx, safe_prompt, target_form))

    print(f"[{os.path.basename(in_path)}] {len(flat)} safe prompts to rewrite.")

    num_changed = 0
    for start in tqdm(range(0, len(flat), batch_size)):
        chunk = flat[start : start + batch_size]
        safe_prompts = [safe for (_, _, safe, _) in chunk]
        target_forms = [target_form for (_, _, _, target_form) in chunk]
        rewritten = matcher.rewrite_batch(
            safe_prompts,
            target_forms,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        for (item_idx, safe_idx, safe_prompt, _), new_text in zip(chunk, rewritten):
            out_data[item_idx]["instr-resp"][safe_idx]["safe_instruction"] = new_text
            if new_text != safe_prompt:
                num_changed += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")
    print(f"Changed {num_changed} / {len(flat)} safe prompts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument(
        "--out_suffix",
        default="_safe_style_matched",
        help="Suffix appended before file extension.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    matcher = StyleMatcher(args.model)
    for name in args.files:
        in_path = name if os.path.isabs(name) else os.path.join(THIS_DIR, name)
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}{args.out_suffix}{ext}"
        process_file(
            in_path=in_path,
            out_path=out_path,
            matcher=matcher,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    print("Done.")


if __name__ == "__main__":
    main()
