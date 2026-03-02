CUDA_VISIBLE_DEVICES=6 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/train_images/train/hatefulMemes" --split "train_hatefulMemes"
CUDA_VISIBLE_DEVICES=6 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/train_images/train/bad_ads" --split "train_bad_ads"
CUDA_VISIBLE_DEVICES=6 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/train_images/train/harm-p" --split "train_harm-p"
CUDA_VISIBLE_DEVICES=6 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/train_images/train/HOD" --split "train_HOD"
CUDA_VISIBLE_DEVICES=6 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/train_images/train/privacyAlert" --split "train_privacyAlert"


