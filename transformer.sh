#!/bin/bash
PARAM_SET=big
DATA_DIR=$HOME/transformer/data
API=original
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
python data_download.py --data_dir=$DATA_DIR

for i in {1..1}
do
MODEL_DIR=$HOME/transformer/model_$API/model_$PARAM_SET

    python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --bleu_source=$DATA_DIR/newstest2014.en --bleu_ref=$DATA_DIR/newstest2014.de train_steps=5000 --steps_between_evals=5000 |& tee /log/$API$i.txt

    python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=$DATA_DIR/newstest2014.en --file_out=translation.en |& tee -a /log/$API$i.txt

    python compute_bleu.py --translation=translation.en --reference=$DATA_DIR/newstest2014.de |& tee -a /log/$API$i.txt
    
done