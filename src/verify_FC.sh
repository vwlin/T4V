#!/bin/bash

model_pth=$1 # network to verify
L=$2 # number of hidden layers
N=$3 # neurons per layer
eps=$4 # epsilon to verify (out of 255)
out_log_pth=$5 # dump log here

timeout_val=60

cd LiRPA_Verify/src
for image_num in {0..199}
do
    timeout $timeout_val python3 -u bab_verification_t4v.py --load "../../${model_pth}" --model mnist_model --num_hidden_layers $L --layer_size $N --batch_size 1 --timeout 3600  --eps $eps --img $image_num >> "../../${out_log_pth}"
    >&2 echo "Processed image $image_num"
    ((image_num++))

    echo "" >> "../../${out_log_pth}"
done