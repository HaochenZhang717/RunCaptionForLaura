import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


# =========================
# PATHS
# =========================

base_model = "/playpen/haochenz/hf_models/Qwen3-VL-8B-Instruct"
lora_checkpoint = "../qwen3vl_grad_diff/0311/final_checkpoint"

forget_json = "/playpen-shared/laura/unlearning/VLGuard/test_forget_image_only_3_sentence.json"
retain_json = "/playpen-shared/laura/unlearning/VLGuard/test_retain_image_only_3_sentence.json"

image_root = "/playpen-shared/laura/unlearning/VLGuard/test_images/test"

output_dir = "./eval_results"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# JSON helpers
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_human_and_gpt_text(conversations):
    user_text = None
    assistant_text = None

    for turn in conversations:
        if turn["from"] == "human":
            user_text = turn["value"]
        elif turn["from"] == "gpt":
            assistant_text = turn["value"]

    if user_text is None:
        user_text = "<image>"
    if assistant_text is None:
        assistant_text = ""

    return user_text, assistant_text


# =========================
# Load model
# =========================

print("Loading processor...")
processor = AutoProcessor.from_pretrained(lora_checkpoint)

print("Loading base model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, lora_checkpoint)

model = model.to(device)
model.eval()

print("Model ready.")


# =========================
# Inference function
# =========================

def run_inference(data, output_path):

    results = []

    for item in tqdm(data):

        image_path = os.path.join(image_root, item["image"])

        _, gt_caption = find_human_and_gpt_text(item["conversations"])

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]

        result = {
            "id": item["id"],
            "image": item["image"],
            "generated_caption": caption,
            "gt_caption": gt_caption,
        }

        results.append(result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", output_path)


# =========================
# Run evaluation
# =========================

print("Loading forget set...")
forget_data = load_json(forget_json)

print("Running forget inference...")
run_inference(
    forget_data,
    os.path.join(output_dir, "forget_predictions.json"),
)


print("Loading retain set...")
retain_data = load_json(retain_json)

print("Running retain inference...")
run_inference(
    retain_data,
    os.path.join(output_dir, "retain_predictions.json"),
)

print("Done.")