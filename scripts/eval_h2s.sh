DATA_PATH=""
TIME_NOW=$(date +%Y%m%d%H%M%S)
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port 29668 \
main_task_retrieval.py --do_eval \
--data_path data_h2 \
--features_path "./How2Sign/RTMpose/Pose_all_24rates/" \
--features_RGB_path "./How2Sign/I3D_features/" \
--output_dir result_eval/eval_h2s \
--signbert \
--init_model ckpts/h2s_best_model.bin \
--fusion_type 'gloss_atten' --rgb_pose_match --rgb_pose_match_loss 0.4 \
--lr 1e-5 --sign_lr 1e-4 \
--max_words 32 --feature_len 64 --max_length_frames 300 \
--slide_windows 16 --windows_stride 1 --original_size_w 256 --original_size_h 256 \
--crop_size 256 --frames_threshold 0.1 --threshold 0.4 \
--batch_size_val 64 \
--datatype h2s_pose --coef_lr 1. --freeze_layer_num 0 \
--linear_patch 2d --sim_header Filip \
--pretrained_clip_name ViT-B/32