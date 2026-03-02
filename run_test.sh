CUDA_VISIBLE_DEVICES=7 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/test_images/test/hatefulMemes" --split "test_hatefulMemes"
CUDA_VISIBLE_DEVICES=7 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/test_images/test/bad_ads" --split "test_bad_ads"
CUDA_VISIBLE_DEVICES=7 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/test_images/test/harm-p" --split "test_harm-p"
CUDA_VISIBLE_DEVICES=7 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/test_images/test/HOD" --split "test_HOD"
CUDA_VISIBLE_DEVICES=7 python run_caption_laura.py --part_id 0 --num_parts 1  --image_folder "/playpen-shared/laura/unlearning/VLGuard/test_images/test/privacyAlert" --split "test_privacyAlert"


