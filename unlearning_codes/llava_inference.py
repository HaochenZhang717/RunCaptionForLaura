import os
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


# =========================
# Dataset (对齐 VLGuardDataset)
# =========================

class VLGuardInferDataset:
    def __init__(self, json_path, image_root=None, multimodal=True):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root) if image_root else None
        self.multimodal = multimodal

        self.data = json.loads(self.json_path.read_text())

    def __len__(self):
        return len(self.data)

    def get_item(self, item):

        # =========================
        # 1. parse question
        # =========================
        question = ""

        if "messages" in item:
            for turn in item["messages"]:
                if turn.get("role") == "user":
                    question = turn.get("content", "")
                    break

        question = (question or "").replace("<image>", "").strip()

        # =========================
        # 2. parse image
        # =========================
        image = None
        image_path = None

        if self.multimodal and "images" in item:
            image_path = item["images"][0]

            # 如果不是绝对路径，用 image_root
            if self.image_root is not None and not os.path.isabs(image_path):
                image_path = os.path.join(self.image_root, image_path)

            image = Image.open(image_path).convert("RGB")

        # =========================
        # 3. metadata
        # =========================
        meta = item.get("metadata", {})

        return {
            "question": question,
            "image": image,
            "image_path": image_path,
            "id": meta.get("id"),
            "source_index": meta.get("source_index"),
            "turn_index": meta.get("turn_index"),
            "safe": meta.get("safe"),
        }

# =========================
# Prompt（完全对齐 training）
# =========================

def build_prompt(mode, question):

    if mode == "multimodal":
        return f"USER: <image>\n{question}\nASSISTANT:"

    elif mode == "image_only":
        return """USER: <image>
Describe the image in no more than 3 sentences.
ASSISTANT:"""

    elif mode == "text_only":
        return f"USER: {question}\nASSISTANT:"

    else:
        raise ValueError("Invalid mode")


# =========================
# Inference
# =========================

def run_inference(args):

    dataset = VLGuardInferDataset(
        json_path=args.input_json,
        image_root=args.image_root,
        multimodal=(args.mode != "text_only"),
    )

    print(f"Loaded {len(dataset)} samples")

    # ===== model =====
    processor = AutoProcessor.from_pretrained(args.model_path)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # LoRA or merged
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print("Loading LoRA...")
        model = PeftModel.from_pretrained(model, args.model_path)

    model = model.to(args.device)
    model.eval()

    results = []

    for item in tqdm(dataset.data):

        sample = dataset.get_item(item)

        prompt = build_prompt(args.mode, sample["question"])
        # print(prompt)
        # breakpoint()
        inputs = processor(
            text=[prompt],
            images=[sample["image"]] if sample["image"] is not None else None,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # trim prompt
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], outputs)
        ]

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        results.append({
            "id": sample["id"],
            "image": sample["image_name"],
            "output": text,
        })

    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, args.output_name)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved to:", out_path)


# =========================
# Args
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--image_root", type=str)

    parser.add_argument("--mode", type=str, choices=[
        "multimodal",
        "image_only",
        "text_only"
    ], required=True)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)