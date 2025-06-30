coco_path=$1
backbone_dir=$2
export CUDA_VISIBLE_DEVICES=$3 && python3 main.py \
	--output_dir logs/DINO/Swin4scale-MS4 -c config/DINO/DINO_4scale_swin_disabilityparking.py --coco_path $coco_path \
	--pretrain_model_path /gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/DINO/swin_backbone_dir/checkpoint0029_4scale_swin.pth \
	--finetune_ignore label_enc.weight class_embed \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir

