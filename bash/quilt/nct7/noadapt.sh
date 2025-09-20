
DATASET="nct"
DATA_DIR="path/to/NCT7k_val/dataset"

BACKBONE="hf-hub:wisdomik/QuiltNet-B-32"
CLIP_TYPE="open_clip"
METHOD="clipartt"
TEMP_DIR="templates/t1.yaml" 

BATCH_SIZE=128
STEPS=10
LEANING_RATE=0.001
TRIALS=1
WORKERS=0


ALL_CORRUPTIONS="original gaussian_noise shot_noise defocus_blur motion_blur brightness contrast stain_light stain_heavy dust air_bubble"

SAVE_DIR="./save/quilt/${DATASET}/noadapt_bs${BATCH_SIZE}"



CUDA_VISIBLE_DEVICES=0 python main.py --clip_type $CLIP_TYPE --seed 42 --dataset $DATASET --data_dir $DATA_DIR --save_dir $SAVE_DIR \
                                      --workers $WORKERS --batch_size $BATCH_SIZE --steps $STEPS --lr $LEANING_RATE --trials $TRIALS \
                                      --corruptions_list $ALL_CORRUPTIONS \
                                      --method $METHOD --backbone $BACKBONE  --K 3