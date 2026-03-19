import os
import sys
import argparse
import math
import json
from datetime import timedelta
from pathlib import Path

sys.path.append("../")
sys.path.append("../../")

try:
    import pandas as pd
except ImportError:
    pd = None

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from tqdm import tqdm

from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    VLGuardDataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
    train_collate_fn_qwen2_vl_multimodal,
    train_collate_fn_qwen2_vl_unimodal,
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


def main(args):
    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    print("Tokenizer Length:", len(tokenizer))

    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    target_modules = find_all_linear_names(model)
    if not target_modules:
        raise RuntimeError("No nn.Linear modules found for LoRA target_modules.")

    vision_target_modules = [
        name for name in target_modules if ("vision_model" in name) or ("vision_tower" in name)
    ]
    projector_target_modules = [
        name for name in target_modules if ("multi_modal_projector" in name) or ("mm_projector" in name)
    ]
    lm_head_target_modules = [name for name in target_modules if name.endswith("lm_head")]
    print(f"LoRA target linear modules: {len(target_modules)}")
    print(f"LoRA target sample: {target_modules[:12]}")
    print(
        "LoRA target breakdown | "
        f"vision: {len(vision_target_modules)}, "
        f"projector: {len(projector_target_modules)}, "
        f"lm_head: {len(lm_head_target_modules)}"
    )

    effective_ddp_find_unused_parameters = args.ddp_find_unused_parameters
    if (
        not effective_ddp_find_unused_parameters
        and (vision_target_modules or projector_target_modules or lm_head_target_modules)
    ):
        print(
            "WARNING: Auto-enabling DDP find_unused_parameters because LoRA targets include "
            "vision/projector/lm_head modules that may be conditionally unused per step.",
            flush=True,
        )
        effective_ddp_find_unused_parameters = True

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )

    print("Injecting LoRA adapters")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_vision = [name for name in trainable_names if ("vision_model" in name) or ("vision_tower" in name)]
    trainable_projector = [
        name for name in trainable_names if ("multi_modal_projector" in name) or ("mm_projector" in name)
    ]
    print(
        "Trainable parameter breakdown | "
        f"vision: {len(trainable_vision)}, "
        f"projector: {len(trainable_projector)}, "
        f"total: {len(trainable_names)}"
    )
    if sum(param.numel() for param in model.parameters() if param.requires_grad) == 0:
        raise RuntimeError("LoRA injection produced 0 trainable parameters.")

    forget_dataset, retain_dataset = build_datasets(args)
    if args.retain_loss_weight != 0 and retain_dataset is None:
        raise ValueError("KL retain loss requires --retain_json or a parquet retain split when retain_loss_weight != 0.")

    train_dataloader, train_dataloader_retain = build_dataloaders(
        args,
        processor,
        forget_dataset,
        retain_dataset,
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=effective_ddp_find_unused_parameters,
        static_graph=args.ddp_static_graph,
        broadcast_buffers=args.ddp_broadcast_buffers,
    )
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=args.ddp_timeout_seconds),
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, process_group_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    rank = accelerator.process_index
    print(
        f"[rank {rank}] Accelerator initialized | "
        f"world_size={accelerator.num_processes} | "
        f"find_unused={effective_ddp_find_unused_parameters} | "
        f"static_graph={args.ddp_static_graph}",
        flush=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * args.num_epochs
    if args.warmup_steps > 0:
        num_warmup_steps = args.warmup_steps
    else:
        num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    prepare_args = [
        model,
        optimizer,
        train_dataloader,
    ]
    if train_dataloader_retain is not None:
        prepare_args.append(train_dataloader_retain)
    prepare_args.append(lr_scheduler)

    print(f"[rank {rank}] Waiting at pre-prepare barrier", flush=True)
    accelerator.wait_for_everyone()
    print(f"[rank {rank}] Calling accelerator.prepare(...)", flush=True)
    prepared = accelerator.prepare(*prepare_args)
    model = prepared[0]
    optimizer = prepared[1]
    train_dataloader = prepared[2]
    offset = 3
    if train_dataloader_retain is not None:
        train_dataloader_retain = prepared[offset]
        lr_scheduler = prepared[offset + 1]
    else:
        lr_scheduler = prepared[offset]
    print(f"[rank {rank}] accelerator.prepare completed", flush=True)

    for epoch in range(args.num_epochs):
        print(f"[rank {rank}] Starting epoch {epoch + 1}/{args.num_epochs}", flush=True)
        model.train()
        total_loss = 0.0
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        retain_multi_iter = None
        if train_dataloader_retain is not None:
            retain_multi_iter = iter(train_dataloader_retain)

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )

        for step, forget_batch in enumerate(progress_bar):
            if step == 0:
                print(f"[rank {rank}] First batch received for epoch {epoch + 1}", flush=True)
            with accelerator.accumulate(model):
                forget_outputs = invoke(forget_batch, model, args.model_family, args.data_mode)
                forget_loss = forget_outputs.loss
                retain_loss = torch.tensor(0.0, device=accelerator.device)

                if retain_multi_iter is not None and args.retain_loss_weight != 0:
                    retain_batch, retain_multi_iter = next_or_restart(
                        retain_multi_iter,
                        train_dataloader_retain,
                    )
                    retain_outputs = invoke(retain_batch, model, args.model_family, args.data_mode)
                    retain_loss = retain_outputs.loss * args.retain_loss_weight

                total_step_loss = (-args.gamma * forget_loss) + retain_loss
                accelerator.backward(total_step_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = total_step_loss.item()
            total_loss += step_loss
            total_forget_loss += forget_loss.item()
            total_retain_loss += retain_loss.item()

            progress_bar.set_postfix(
                {
                    "step_loss": step_loss,
                    "forget_ce": forget_loss.item(),
                    "retain_ce": retain_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        avg_loss = total_loss / len(train_dataloader)
        avg_forget = total_forget_loss / len(train_dataloader)
        avg_retain = total_retain_loss / len(train_dataloader)
        print(
            f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f} | "
            f"Forget CE: {avg_forget:.4f} | Retain CE: {avg_retain:.4f}"
        )

        if args.save_per_epoch:
            epoch_save_dir = Path(args.save_dir) / f"checkpoint-epoch-{epoch + 1}"
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                processor=processor,
                output_dir=epoch_save_dir,
                merge=False,
            )

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        processor=processor,
        output_dir=args.save_dir,
        merge=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL unlearning for multimodal models")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Model path")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default=None, help="Directory of the parquet split dataset")
    parser.add_argument("--forget_json", type=str, default=None, help="Path to a multimodal forget JSON file")
    parser.add_argument("--retain_json", type=str, default=None, help="Path to a multimodal retain JSON file")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for relative image paths")
    parser.add_argument(
        "--data_mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_only"],
        help="Whether to train with image-text inputs or text-only inputs.",
    )
    parser.add_argument("--forget_split_ratio", type=int, default=15, help="Forget ratio for parquet splits")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=1.0, help="Scale for the forget loss term")
    parser.add_argument("--retain_loss_weight", type=float, default=1.0, help="Scale for the retain KL term")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio over total train steps")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps (overrides warmup_ratio if > 0)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--allow_multi_worker_collate",
        action="store_true",
        help="Allow num_workers>0 even though processor-in-collate may hang on some setups.",
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        action="store_true",
        help="Use persistent workers when dataloader_num_workers>0.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor when dataloader_num_workers>0 (<=0 disables explicit setting).",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP find_unused_parameters.",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="Enable DDP static_graph optimization when graph is stable.",
    )
    parser.add_argument(
        "--ddp_broadcast_buffers",
        action="store_true",
        help="Enable DDP broadcast_buffers.",
    )
    parser.add_argument(
        "--ddp_timeout_seconds",
        type=int,
        default=600,
        help="Distributed process-group timeout in seconds.",
    )
    parser.add_argument("--save_per_epoch", action="store_true", help="Save an adapter checkpoint after each epoch")
    args = parser.parse_args()
    main(args)