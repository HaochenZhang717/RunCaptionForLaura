#!/bin/bash

python llava_infer.py \
  --base_model /playpen/haochenz/hf_models/llava \
  --model_path /playpen-shared/haochenz/your_checkpoint \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_input.json \
  --image_root /playpen-shared/laura/unlearning/VLGuard/test_images/test \
  --mode multimodal \
  --output_name multimodal.json


#!/bin/bash

python llava_infer.py \
  --base_model /playpen/haochenz/hf_models/llava \
  --model_path /playpen-shared/haochenz/your_checkpoint \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_input_text_only.json \
  --mode text_only \
  --output_name text_only.json


#!/bin/bash

python llava_infer.py \
  --base_model /playpen/haochenz/hf_models/llava \
  --model_path /playpen-shared/haochenz/your_checkpoint \
  --input_json /playpen-shared/laura/unlearning/safety/vlguard_test_qwen_captioning.json \
  --image_root /playpen-shared/laura/unlearning/VLGuard/test_images/test \
  --mode image_only \
  --output_name image_only.json