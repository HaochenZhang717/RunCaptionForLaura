import os
import sys
import argparse
import math
from datetime import timedelta
from pathlib import Path

sys.path.append("../")
sys.path.append("../../")

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from tqdm import tqdm

# ===== reuse KL.py components =====
from KL import (
    load_model_and_processor,
    build_datasets,
    build_dataloaders,
    invoke,
    find_all_linear_names,
)

from peft import LoraConfig, get_peft_model, PeftModel


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