"""
Paraphrase the human-side questions / instructions in the three JSON files
using an open-source text LLM (default: Qwen2.5-7B-Instruct).

Input files (all under this directory):
    - test.json
    - train_forget.json
    - train_retain.json

Output files (same directory), where <form> is "question" or "instruction":
    - test_rephrased_<form>.json
    - train_forget_rephrased_<form>.json
    - train_retain_rephrased_<form>.json

The --target_form flag picks the unified form: every human text is
rewritten as either an interrogative question OR an imperative instruction.

File formats:
    test.json:
        item["instr-resp"] is a list of dicts, each has either
        "safe_instruction" or "unsafe_instruction" (+ "response").

    train_forget.json / train_retain.json:
        item["conversations"] is a list of {"from", "value"}. The
        human turn's value starts with "<image>" followed by the question.

Only the question / instruction text is paraphrased. Responses, ids,
image paths, and the "<image>" tag are left untouched.
"""

import argparse
import copy
import json
import os
import re
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILES = [
    "test.json",
    "train_forget.json",
    "train_retain.json",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPTS = {
    "question": (
        "You rewrite user inputs about an image into a single INTERROGATIVE "
        "QUESTION. The input may be a question or an imperative instruction; "
        "either way, your output MUST be a natural-sounding question that "
        "ends with a question mark ('?') and preserves the original meaning "
        "and intent. Do not turn it into a command. Do not add new content. "
        "Output ONLY the rewritten question, with no quotes, no prefix, "
        "and no explanation."
    ),
    "instruction": (
        "You rewrite user inputs about an image into a single IMPERATIVE "
        "INSTRUCTION. The input may be a question or an instruction; either "
        "way, your output MUST be a directive command (starting with a verb "
        "such as 'Describe', 'Explain', 'List', 'Tell me', 'Provide', etc.) "
        "that preserves the original meaning and intent. Do not phrase it as "
        "a question and do not end with a question mark. Do not add new "
        "content. Output ONLY the rewritten instruction, with no quotes, "
        "no prefix, and no explanation."
    ),
}

USER_TEMPLATES = {
    "question": "Rewrite the following as a single interrogative question:\n\n{text}",
    "instruction": "Rewrite the following as a single imperative instruction:\n\n{text}",
}

IMAGE_TAG = "<image>"


# =========================
# Extract / inject text
# =========================

def strip_image_tag(text: str) -> Tuple[str, bool]:
    """Return (text_without_tag, had_tag). Matches a leading <image> with
    optional whitespace after it."""
    m = re.match(r"\s*<image>\s*", text)
    if m:
        return text[m.end():], True
    return text, False


def collect_texts_from_item(item: dict) -> List[Tuple[str, str, bool]]:
    """
    Return a list of (path_key, original_text, had_image_tag) tuples
    describing every human text inside the item. path_key encodes how
    to put the paraphrased text back.
    """
    slots = []

    # test.json style
    if "instr-resp" in item:
        for i, pair in enumerate(item["instr-resp"]):
            for key in ("safe_instruction", "unsafe_instruction"):
                if key in pair and isinstance(pair[key], str):
                    stripped, had_tag = strip_image_tag(pair[key])
                    slots.append((f"instr-resp[{i}].{key}", stripped, had_tag))

    # train_*.json style
    if "conversations" in item:
        for i, turn in enumerate(item["conversations"]):
            if turn.get("from") == "human" and isinstance(turn.get("value"), str):
                stripped, had_tag = strip_image_tag(turn["value"])
                slots.append((f"conversations[{i}].value", stripped, had_tag))

    return slots


def set_by_path(item: dict, path_key: str, new_text: str, had_tag: bool) -> None:
    """Write new_text back into item at path_key, re-adding <image> if needed."""
    final = (IMAGE_TAG + new_text) if had_tag else new_text

    m = re.match(r"(\w[\w\-]*)\[(\d+)\]\.(\w+)", path_key)
    if not m:
        raise ValueError(f"Unrecognized path_key: {path_key}")
    top, idx, leaf = m.group(1), int(m.group(2)), m.group(3)
    item[top][idx][leaf] = final


# =========================
# Model wrapper
# =========================

class Paraphraser:
    def __init__(self, model_path: str, target_form: str,
                 device: str = None, dtype=torch.bfloat16):
        if target_form not in SYSTEM_PROMPTS:
            raise ValueError(
                f"target_form must be one of {list(SYSTEM_PROMPTS)}, got {target_form!r}"
            )
        self.target_form = target_form
        self.system_prompt = SYSTEM_PROMPTS[target_form]
        self.user_template = USER_TEMPLATES[target_form]

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

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(text=text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def paraphrase_batch(
        self,
        texts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        prompts = [self._build_prompt(t) for t in texts]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        trimmed = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        return [clean_output(d) for d in decoded]


def clean_output(s: str) -> str:
    s = s.strip()
    # Remove leading/trailing matching quotes if present.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'", "“", "”"):
        s = s[1:-1].strip()
    return s


# =========================
# Process one file
# =========================

def process_file(in_path: str, out_path: str, para: Paraphraser, batch_size: int):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = copy.deepcopy(data)

    # Gather every (item_idx, path_key, text, had_tag).
    flat = []
    for i, item in enumerate(out_data):
        for path_key, text, had_tag in collect_texts_from_item(item):
            flat.append((i, path_key, text, had_tag))

    print(f"[{os.path.basename(in_path)}] {len(flat)} texts to paraphrase.")

    for start in tqdm(range(0, len(flat), batch_size)):
        chunk = flat[start : start + batch_size]
        texts = [t for (_, _, t, _) in chunk]
        rephrased = para.paraphrase_batch(texts)
        for (i, path_key, _, had_tag), new_text in zip(chunk, rephrased):
            set_by_path(out_data[i], path_key, new_text, had_tag)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Path or HF id of the instruct LLM used for paraphrasing.")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="JSON files (relative to this script's directory).")
    ap.add_argument("--target_form", choices=["question", "instruction"],
                    required=True,
                    help="Unify every human text into this form.")
    ap.add_argument("--out_suffix", default=None,
                    help="Suffix appended before the extension. "
                         "Defaults to '_rephrased_<target_form>'.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    para = Paraphraser(args.model, target_form=args.target_form)
    # Bind sampling params onto the callable for convenience.
    orig_paraphrase = para.paraphrase_batch

    def paraphrase_batch(texts):
        return orig_paraphrase(
            texts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    para.paraphrase_batch = paraphrase_batch

    suffix = args.out_suffix or f"_rephrased_{args.target_form}"
    for name in args.files:
        in_path = name if os.path.isabs(name) else os.path.join(THIS_DIR, name)
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}{suffix}{ext}"
        process_file(in_path, out_path, para, args.batch_size)

    print("Done.")


if __name__ == "__main__":
    main()
