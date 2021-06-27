#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$HOME/ModelGarden/models-r1.13.0
#PARAM_SET=big
DATA_DIR=$HOME/imagenet/data
API=original
#VOCAB_FILE=$DATA_DIR/vocab.ende.32768
#python data_download.py --data_dir=$DATA_DIR

for i in {1..1}
do
MODEL_DIR=$HOME/imagenet/model_$API/

    python imagenet_main.py --data_dir=$DATA_DIR --num_gpus=1 --model_dir=$MODEL_DIR \
     --batch_size=32 |& tee $API$i.txt

done
