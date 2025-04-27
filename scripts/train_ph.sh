DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node=8 --master_port 29558 \
main_task_retrieval.py --do_train --num_thread_reader=64 \
--epochs=200 --batch_size=128 --n_display=10 \
--data_path data_ph \
--features_path "./PHOENIX-2014-T/features/RTM_Keypoints/" \
--features_RGB_path "./PHOENIX-2014-T/features/I3D_features/" \
--output_dir result_train/ph \
--signbert --init_sign_model ckpt/pretrain_signbert.pth \
--fusion_type 'gloss_atten' --rgb_pose_match --rgb_pose_match_loss 0.4 \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.4 \
--batch_size_val 64 \
--datatype ph_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip \
--pretrained_clip_name ViT-B/32
