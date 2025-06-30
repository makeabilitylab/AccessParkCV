coco_path=$1
python3 main.py \
	--output_dir logs/DINO/dp-R50-MS4-largerdataset -c config/DINO/DINO_5scale_disabilityparking_largerdataset.py \
	--coco_path $coco_path \
	--finetune_ignore label_enc.weight class_embed \
	--pretrain_model_path /gscratch/makelab/jaredhwa/DisabilityParking/cv/models/locator/DINO/checkpoint0031_5scale.pth \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
