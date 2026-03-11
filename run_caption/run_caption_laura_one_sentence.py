import os
import json
import glob
import math
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



# ==============================
# Config
# ==============================
model_name = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"



max_new_tokens = 1024

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28





TIME_SERIES_CAPTION_PROMPT = r"""
You are an image describer.

You are given an image.

Your task is to describe the image in 1 sentences. Do not perform any interpretation or speculation about the image.

Now generate the description.
""".strip()


# ==============================
# Load model + processor
# ==============================
print(f"[INFO] Loading model: {model_name}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
    device_map=None,
).eval().to(device)

processor = AutoProcessor.from_pretrained(model_name)

print("[INFO] Model loaded successfully.")


def load_existing_image_set(jsonl_path):
    existing = set()
    if not os.path.exists(jsonl_path):
        return existing

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "image" in obj:
                    existing.add(obj["image"])
            except Exception:
                continue
    return existing


@torch.no_grad()
def caption_one_image(image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": TIME_SERIES_CAPTION_PROMPT
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,

    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def split_range(total, part_id, num_parts):
    """
    Split [0, total) into num_parts nearly-equal contiguous chunks.
    Return (start, end) for chunk part_id.
    """
    chunk_size = math.ceil(total / num_parts)
    start = part_id * chunk_size
    end = min((part_id + 1) * chunk_size, total)
    return start, end



def collect_image_files(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_id", type=int, default=0,
                        help="which split to run (0-based)")
    parser.add_argument("--num_parts", type=int, default=4,
                        help="total number of splits")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--split", type=str, required=True,)


    args = parser.parse_args()

    output_jsonl = f"{args.split}_image_caps_{args.part_id}_{args.num_parts}.jsonl"
    error_jsonl = f"{args.split}_image_caps_errors_{args.part_id}_{args.num_parts}.jsonl"

    assert 0 <= args.part_id < args.num_parts, "part_id must be in [0, num_parts)"

    png_files = collect_image_files(args.image_folder)
    total = len(png_files)

    start_idx, end_idx = split_range(total, args.part_id, args.num_parts)
    png_files = png_files[start_idx:end_idx]

    print(f"[INFO] Total PNG files: {total}")
    print(f"[INFO] Running part {args.part_id}/{args.num_parts} -> range [{start_idx}, {end_idx})")
    print(f"[INFO] This part contains {len(png_files)} images")

    processed_images = load_existing_image_set(output_jsonl)
    print(f"[INFO] Already processed {len(processed_images)} images (resume enabled).")

    fout = open(output_jsonl, "a", encoding="utf-8")
    ferr = open(error_jsonl, "a", encoding="utf-8")

    num_success = 0
    num_fail = 0

    for img_path in tqdm(png_files, desc=f"Captioning part {args.part_id}"):
        img_name = os.path.basename(img_path)

        if img_name in processed_images:
            continue

        try:
            caption = caption_one_image(img_path)
            print(caption)
            record = {
                "image": img_name,
                "image_path": img_path,
                "caption": caption,
                "model": model_name,
                "part_id": args.part_id,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            num_success += 1

        except Exception as e:
            err_record = {
                "image": img_name,
                "image_path": img_path,
                "error": str(e),
                "part_id": args.part_id,
            }
            ferr.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            ferr.flush()
            num_fail += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fout.close()
    ferr.close()

    print(f"[DONE] Success: {num_success}, Failed: {num_fail}")
    print(f"[DONE] Output saved to: {output_jsonl}")
    print(f"[DONE] Errors saved to: {error_jsonl}")


if __name__ == "__main__":
    main()