import os
import json
import math
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import nested_detach

# =========================================================
# Optional helpers for custom trainer internals
# =========================================================
from transformers.utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =========================================================
# Placeholder if you don't use external evaluators
# =========================================================
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


# =========================================================
# Dummy KL helper
# Replace this with your own `from trainer.utils import compute_kl_divergence`
# if you already have it implemented.
# =========================================================
def compute_kl_divergence(model, ref_model, batch):
    """
    batch: dict with input_ids, attention_mask, labels, pixel_values, image_grid_thw, ...
    Only computes KL on valid label positions (labels != -100).
    """
    with torch.no_grad():
        ref_outputs = ref_model(**batch)
        ref_logits = ref_outputs.logits

    outputs = model(**batch)
    logits = outputs.logits

    # shift if causal LM loss is token t predicting t+1
    # Here we align same logits positions and then mask by labels != -100
    labels = batch["labels"]

    # Flatten
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    ref_logits_flat = ref_logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    mask = labels_flat != -100
    if mask.sum() == 0:
        kl = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return kl, outputs

    log_probs = torch.log_softmax(logits_flat[mask], dim=-1)
    ref_probs = torch.softmax(ref_logits_flat[mask], dim=-1)

    kl = torch.nn.functional.kl_div(
        log_probs,
        ref_probs,
        reduction="batchmean",
        log_target=False,
    )
    return kl, outputs


# =========================================================
# Base FinetuneTrainer
# =========================================================
class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning(
                    "Custom evaluator can be run only when a single accelerator process is running."
                )
                return {}

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)

            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.processing_class,
                }
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


# =========================================================
# UnlearnTrainer
# =========================================================
class UnlearnTrainer(FinetuneTrainer):
    def _prepare_deepspeed(self, model):
        from copy import deepcopy
        import deepspeed

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config,
                    "keys_to_ignore_at_inference",
                    ["past_key_values"],
                )
            else:
                ignore_keys = []

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if isinstance(logits, (tuple, list)) and len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


# =========================================================
# GradDiff trainer
# =========================================================
class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        import copy

        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0

        if self.retain_loss_type == "NLL":
            retain_loss = retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss = kl_loss
        else:
            raise NotImplementedError(f"{self.retain_loss_type} not implemented for retain set")

        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss


# =========================================================
# Data utilities
# =========================================================
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


def resolve_image_path(image_path, image_root=None, json_dir=None):
    if os.path.isabs(image_path):
        return image_path
    if image_root is not None:
        candidate = os.path.join(image_root, image_path)
        if os.path.exists(candidate):
            return candidate
    if json_dir is not None:
        candidate = os.path.join(json_dir, image_path)
        if os.path.exists(candidate):
            return candidate
    return image_path


class VLGuardSingleDataset(Dataset):
    def __init__(self, json_path, image_root=None):
        self.json_path = json_path
        self.json_dir = os.path.dirname(json_path)
        self.image_root = image_root
        self.data = load_json(json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        _, assistant_text = find_human_and_gpt_text(item["conversations"])
        image_path = resolve_image_path(
            item["image"],
            image_root=self.image_root,
            json_dir=self.json_dir,
        )
        return {
            "id": item["id"],
            "image_path": image_path,
            "answer": assistant_text,
        }


class PairedForgetRetainDataset(Dataset):
    """
    Each item returns one forget sample + one retain sample.
    Length = max(len(forget), len(retain))
    """
    def __init__(self, forget_dataset, retain_dataset):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.length = max(len(forget_dataset), len(retain_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        forget_item = self.forget_dataset[idx % len(self.forget_dataset)]
        retain_item = self.retain_dataset[idx % len(self.retain_dataset)]
        return {
            "forget": forget_item,
            "retain": retain_item,
        }


# =========================================================
# Qwen3-VL batch builder
# =========================================================

PROMPT = """
You are given an image.

Your task is to describe the image in no more than 3 sentences. Do not perform any interpretation or speculation about the image.

Now generate the description."""

class Qwen3VLCollatorForGradDiff:
    def __init__(
        self,
        processor,
        max_length=1024,
        image_min_pixels=None,
        image_max_pixels=None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def _build_messages(self, answer_text):
        # training sample format:
        # user: image + "Describe this image."
        # assistant: target caption
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text},
                ],
            },
        ]

    def _build_prompt_only_messages(self):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

    def _encode_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_texts = []

        for sample in samples:
            image = self._load_image(sample["image_path"])
            images.append(image)

            full_messages = self._build_messages(sample["answer"])
            prompt_messages = self._build_prompt_only_messages()

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            # truncation=True,
            # max_length=self.max_length,
            return_tensors="pt",
        )

        with self.processor.as_target_processor() if hasattr(self.processor, "as_target_processor") else DummyContext():
            prompt_batch = self.processor(
                text=prompt_texts,
                images=images,
                padding=True,
                # truncation=True,
                # max_length=self.max_length,
                return_tensors="pt",
            )

        labels = batch["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # mask prompt tokens, only optimize assistant response tokens
        for i in range(labels.size(0)):
            prompt_len = int(prompt_batch["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        forget_samples = [f["forget"] for f in features]
        retain_samples = [f["retain"] for f in features]

        forget_batch = self._encode_batch(forget_samples)
        retain_batch = self._encode_batch(retain_samples)

        return {
            "forget": forget_batch,
            "retain": retain_batch,
        }


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


# =========================================================
# Main
# =========================================================
def main():
    model_name = "/playpen/haochenz/hf_models/Qwen3-VL-8B-Instruct"

    forget_json = "/playpen-shared/laura/unlearning/VLGuard/train_forget_image_only_3_sentence.json"
    retain_json = "/playpen-shared/laura/unlearning/VLGuard/train_retain_image_only_3_sentence.json"

    # very likely images are under this root
    # image_root = "/playpen-shared/laura/unlearning/VLGuard"
    image_root = "/playpen-shared/laura/unlearning/VLGuard/train_images/train"

    output_dir = "../qwen3vl_grad_diff/0312"

    # -------------------------
    # load model / processor
    # -------------------------
    processor = AutoProcessor.from_pretrained(model_name)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",   # remove if env doesn't support it
    )

    # optional but often helpful for training memory
    model.config.use_cache = False
    model.enable_input_require_grads()



    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # -------------------------
    # datasets
    # -------------------------
    forget_dataset = VLGuardSingleDataset(
        json_path=forget_json,
        image_root=image_root,
    )
    retain_dataset = VLGuardSingleDataset(
        json_path=retain_json,
        image_root=image_root,
    )

    train_dataset = PairedForgetRetainDataset(
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
    )

    data_collator = Qwen3VLCollatorForGradDiff(
        processor=processor,
        max_length=1024,
    )

    # -------------------------
    # training args
    # -------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=5,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,   # very important for custom collator inputs
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    # -------------------------
    # trainer
    # -------------------------
    trainer = GradDiff(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=processor,   # your FinetuneTrainer uses this as tokenizer
        data_collator=data_collator,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",       # or "KL"
    )

    trainer.train()

    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
    processor.save_pretrained(os.path.join(output_dir, "final_checkpoint"))


if __name__ == "__main__":
    main()