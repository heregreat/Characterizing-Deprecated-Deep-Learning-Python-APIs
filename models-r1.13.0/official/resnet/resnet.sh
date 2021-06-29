#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$HOME/ModelGarden/models-r1.13.0
#PARAM_SET=big
DATA_DIR=/root/imagenet/data
API=max_pooling2d
#VOCAB_FILE=$DATA_DIR/vocab.ende.32768
#python data_download.py --data_dir=$DATA_DIR

for i in {1..1}
do
MODEL_DIR=/root/imagenet/model_$API$i/

    python imagenet_main.py --data_dir=$DATA_DIR --num_gpus=1 --model_dir=$MODEL_DIR \
     --batch_size=32 --max_train_steps=5000 --train_epochs=1 |& tee $API$i.txt

done
mv *.txt $API
