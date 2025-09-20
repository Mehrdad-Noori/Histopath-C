
DATASET="skin"
DATA_DIR="path/to/skin/dataset"

BACKBONE="hf-hub:wisdomik/QuiltNet-B-32"
CLIP_TYPE="open_clip"
METHOD="latte"
TEMP_DIR="templates/t25.yaml" 

BATCH_SIZE=128
STEPS=10
LEANING_RATE=0.001
TRIALS=2
WORKERS=0
AVG_TYPE="loss_avg"

## LORA PARAMS
LORA_PARAMS="q k v o"
LORA_RANK=2
LORA_ALPHA=1
LORA_DROPOUT=0.0
LORA_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11"

ALL_CORRUPTIONS="original gaussian_noise shot_noise defocus_blur motion_blur brightness contrast stain_light stain_heavy dust air_bubble"


LORA_PARAMS_DIR="${LORA_PARAMS// /_}"

SAVE_DIR="./save/quilt/${DATASET}/${METHOD}_Lora_LN_bs${BATCH_SIZE}_lr${LEANING_RATE}_s${STEPS}_t${loss_avg}_P${LORA_PARAMS_DIR}_a${LORA_ALPHA}_r${LORA_RANK}"

CUDA_VISIBLE_DEVICES=0  python main.py  --clip_type $CLIP_TYPE --seed 42 --dataset $DATASET --data_dir $DATA_DIR --save_dir $SAVE_DIR \
                                        --workers $WORKERS --batch_size $BATCH_SIZE --steps $STEPS --lr $LEANING_RATE --trials $TRIALS \
                                        --corruptions_list  $ALL_CORRUPTIONS \
                                        --method $METHOD --backbone $BACKBONE --temp_dir $TEMP_DIR  --adapt --avg_type $AVG_TYPE \
                                        --lora_ln --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --lora_rank $LORA_RANK --lora_params $LORA_PARAMS --lora_layers $LORA_LAYERS