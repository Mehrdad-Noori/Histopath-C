

DATASET="mhist"
DATA_DIR="path/to/mhist/dataset"

BACKBONE="hf-hub:wisdomik/QuiltNet-B-32"
CLIP_TYPE="open_clip"
METHOD="tpt"
TEMP_DIR="templates/t1.yaml" 

BATCH_SIZE=128
STEPS=10
LEANING_RATE=0.001
TRIALS=2
WORKERS=0


ALL_CORRUPTIONS="original gaussian_noise shot_noise defocus_blur motion_blur brightness contrast stain_light stain_heavy dust air_bubble"

SAVE_DIR="./save/quilt/${DATASET}/${METHOD}_bs${BATCH_SIZE}_lr${LEANING_RATE}_s${STEPS}"



CUDA_VISIBLE_DEVICES=0 python main.py --clip_type $CLIP_TYPE --save_dir $SAVE_DIR --seed 42 --dataset $DATASET --data_dir $DATA_DIR \
                                      --workers $WORKERS --batch_size $BATCH_SIZE --steps $STEPS --lr $LEANING_RATE --trials $TRIALS \
                                      --corruptions_list $ALL_CORRUPTIONS \
                                      --method $METHOD --backbone $BACKBONE --temp_dir $TEMP_DIR --adapt