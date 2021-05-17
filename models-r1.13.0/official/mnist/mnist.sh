#!/bin/bash
export PYTHONPATH="$PYTHONPATH:/root/ModelGarden/models-r1.13.0"
for i in {1..20}
do
rm -rf /tmp/mnist_model
python3 mnist.py > $i.txt
done

