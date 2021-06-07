#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$HOME/ModelGarden/models-r2.1.0
DATA_DIR=$HOME/resnet
API=v1.nn.soft
#python data_download.py --data_dir=$DATA_DIR |& tee download.txt

for i in {1..20}
do
MODEL_DIR=$HOME/resnet/model_$API/model_$PARAM_SET

    python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --train_steps=5000 --steps_between_evals=5000 |& tee $API$i.txt

done
