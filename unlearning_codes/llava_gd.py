import os
import sys
import argparse
import math
from datetime import timedelta
from pathlib import Path
sys.path.append("../")
sys.path.append("../../")

from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    VLGuardDataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
    train_collate_fn_qwen2_vl_multimodal,
    train_collate_fn_qwen2_vl_unimodal,
)


# ===== reuse KL.py components =====
# from KL import (
#     load_model_and_processor,
#     build_datasets,
#     build_dataloaders,
#     invoke,
#     find_all_linear_names,
# )

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)


def find_all_linear_names(model):
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_module_names.append(name)
    return sorted(linear_module_names)


def resolve_model_family(*identifiers):
    for identifier in identifiers:
        if not identifier:
            continue

        normalized = str(identifier).strip()
        lowered = normalized.lower()
        if "llava" in lowered:
            return "llava"
        if "qwen" in lowered:
            return "qwen"

        config_path = Path(normalized) / "config.json"
        if not config_path.is_file():
            continue

        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        metadata = " ".join(
            str(value)
            for value in (
                config.get("model_type"),
                config.get("text_config", {}).get("model_type")
                if isinstance(config.get("text_config"), dict)
                else None,
                " ".join(config.get("architectures", [])),
            )
            if value
        ).lower()
        if "llava" in metadata:
            return "llava"
        if "qwen" in metadata:
            return "qwen"

    raise ValueError(
        "Model ID not recognized or not supported. "
        "Expected an LLAVA or Qwen2-VL repo/path, or a config.json that identifies one."
    )


def load_model_and_processor(args):
    model_family = resolve_model_family(args.model_id, args.vanilla_dir)
    args.model_family = model_family

    if model_family == "llava":
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    elif model_family == "qwen":
        print("Loading Qwen2-VL model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    processor.tokenizer.padding_side = "right"
    special_tokens = {}
    if processor.tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<pad>"
    if model_family == "llava" and "<image>" not in processor.tokenizer.get_vocab():
        special_tokens["additional_special_tokens"] = ["<image>"]
    if special_tokens:
        processor.tokenizer.add_special_tokens(special_tokens)

    return model, processor


def invoke(batch, model, model_family, data_mode):
    if model_family not in {"llava", "qwen"}:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    input_ids, attention_mask, pixel_values, labels = batch
    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if data_mode == "multimodal":
        model_kwargs["pixel_values"] = pixel_values
    return model(**model_kwargs)


def build_datasets(args):
    multimodal = args.data_mode == "multimodal"
    if args.forget_json:
        forget_dataset = VLGuardDataset(
            json_path=args.forget_json,
            image_root=args.image_root,
            multimodal=multimodal,
        )
        retain_dataset = None
        if args.retain_json:
            retain_dataset = VLGuardDataset(
                json_path=args.retain_json,
                image_root=args.image_root,
                multimodal=multimodal,
            )

        print(f"Loaded forget JSON from {args.forget_json}: {len(forget_dataset)} samples")
        if retain_dataset is not None:
            print(f"Loaded retain JSON from {args.retain_json}: {len(retain_dataset)} samples")
        return forget_dataset, retain_dataset

    if not args.data_split_dir:
        raise ValueError("Provide either --forget_json/--retain_json or --data_split_dir.")
    if pd is None:
        raise ImportError("pandas is required for parquet split training. Use JSON inputs or install pandas.")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder:", forget_folder)
    print("Retain Folder:", retain_folder)

    forget_parquet_file = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, "train-00000-of-00001.parquet")
    df_forget = pd.read_parquet(forget_parquet_file)
    df_retain = pd.read_parquet(retain_parquet_file)

    if args.data_mode == "multimodal":
        forget_dataset = Muitimodal_Dataset(df=df_forget, mode=f"forget_{args.forget_split_ratio}")
        retain_dataset = Muitimodal_Dataset(df=df_retain, mode=f"retain_{100 - args.forget_split_ratio}")
    else:
        forget_dataset = Unimodal_Dataset(df=df_forget, mode=f"forget_{args.forget_split_ratio}")
        retain_dataset = Unimodal_Dataset(df=df_retain, mode=f"retain_{100 - args.forget_split_ratio}")
    return forget_dataset, retain_dataset


def build_dataloaders(args, processor, forget_dataset, retain_dataset):
    effective_num_workers = args.dataloader_num_workers
    if effective_num_workers > 0 and not args.allow_multi_worker_collate:
        print(
            "WARNING: forcing dataloader_num_workers=0 because collate uses "
            "processor/tokenizer and can hang with multi-worker loading. "
            "Set --allow_multi_worker_collate to opt in."
        )
        effective_num_workers = 0

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": effective_num_workers,
        "pin_memory": True,
    }
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = args.dataloader_persistent_workers
        if args.dataloader_prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

    if args.model_family == "llava":
        collate_fn = (
            (lambda x: train_collate_fn_llava_multimodal(x, processor, args))
            if args.data_mode == "multimodal"
            else (lambda x: train_collate_fn_llava_unimodal(x, processor, args))
        )
    elif args.model_family == "qwen":
        collate_fn = (
            (lambda x: train_collate_fn_qwen2_vl_multimodal(x, processor, args))
            if args.data_mode == "multimodal"
            else (lambda x: train_collate_fn_qwen2_vl_unimodal(x, processor, args))
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    forget_loader = DataLoader(forget_dataset, collate_fn=collate_fn, **loader_kwargs)
    retain_loader = None
    if retain_dataset is not None:
        retain_loader = DataLoader(retain_dataset, collate_fn=collate_fn, **loader_kwargs)

    return forget_loader, retain_loader


def next_or_restart(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def save_checkpoint(accelerator, model, processor, output_dir, merge=False):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if isinstance(unwrapped_model, PeftModel) and merge:
        model_to_save = unwrapped_model.merge_and_unload()
    else:
        model_to_save = unwrapped_model

    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_to_save.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")


# =========================
# save
# =========================
def save_checkpoint(accelerator, model, processor, output_dir, merge=False):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if isinstance(unwrapped_model, PeftModel) and merge:
        model_to_save = unwrapped_model.merge_and_unload()
    else:
        model_to_save = unwrapped_model

    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_to_save.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Saved to {output_dir}")


# =========================
# main
# =========================
def main(args):

    # ===== model =====
    model, processor = load_model_and_processor(args)

    model.config.use_cache = False
    model.resize_token_embeddings(len(processor.tokenizer))

    # ===== LoRA (same as KL.py) =====
    target_modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ===== dataset =====
    forget_dataset, retain_dataset = build_datasets(args)

    train_loader, retain_loader = build_dataloaders(
        args, processor, forget_dataset, retain_dataset
    )

    # ===== accelerator =====
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    process_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=args.ddp_timeout_seconds)
    )

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, process_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # ===== optimizer =====
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs

    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, retain_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, retain_loader, scheduler
    )

    # ===== training =====
    for epoch in range(args.num_epochs):

        model.train()
        retain_iter = iter(retain_loader) if retain_loader else None

        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)

        for step, forget_batch in enumerate(progress_bar):

            with accelerator.accumulate(model):

                # ===== forget =====
                forget_outputs = invoke(forget_batch, model, args.model_family, args.data_mode)
                forget_loss = forget_outputs.loss

                # ===== retain =====
                retain_loss = torch.tensor(0.0, device=accelerator.device)

                if retain_iter and args.retain_loss_weight > 0:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        retain_batch = next(retain_iter)

                    retain_outputs = invoke(retain_batch, model, args.model_family, args.data_mode)
                    retain_loss = retain_outputs.loss

                # ===== GradDiff loss =====
                loss = (-args.gamma * forget_loss) + (args.retain_loss_weight * retain_loss)

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({
                "loss": loss.item(),
                "forget": forget_loss.item(),
                "retain": retain_loss.item()
            })

        # ===== save =====
        if args.save_per_epoch:
            save_checkpoint(
                accelerator,
                model,
                processor,
                os.path.join(args.save_dir, f"epoch_{epoch}")
            )

    save_checkpoint(accelerator, model, processor, args.save_dir, merge=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # === same args as KL.py ===
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--vanilla_dir", type=str)
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--forget_json", type=str)
    parser.add_argument("--retain_json", type=str)
    parser.add_argument("--image_root", type=str)

    parser.add_argument("--data_mode", type=str, default="multimodal")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument(
        "--allow_multi_worker_collate",
        action="store_true",
        help="Allow multi-worker dataloader even if collate uses tokenizer"
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--retain_loss_weight", type=float, default=1.0)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--ddp_timeout_seconds", type=int, default=600)

    parser.add_argument("--save_per_epoch", action="store_true")

    args = parser.parse_args()
    main(args)

    # base model: llava-hf/llava-1.5-7b-hf
    # Lora: rank-32 alpha-16 lora dropout 0.05
    # lr: 1e-5
    # global batch size: 4
    # weight decay 1e-2
    # num_epochs: 4