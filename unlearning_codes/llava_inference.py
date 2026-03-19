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

        question = ""
        image = None

        if "conversations" in item:
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    question = turn["value"]

        question = (question or "").replace("<image>", "").strip()

        if self.multimodal:
            # print(f"self.image_root: {self.image_root}")
            # print(f"item['images']: {item['images']}")
            # print(f"item.keys(): {item.keys()}")
            # image_path = os.path.join(self.image_root, item["images"])
            image_path = item["images"][0]
            image = Image.open(image_path).convert("RGB")

        return {
            "question": question,
            "image": image,
            "image_name": item.get("image"),
            "id": item.get("id"),
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

        breakpoint()
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