#!/bin/bash

model_pth=$1 # network to verify
eps=$2 # epsilon to verify (out of 255)
out_log_pth=$3 # dump log here

timeout_val=120

cd LiRPA_Verify/src
for image_num in {0..199}
do
    timeout $timeout_val python3 -u bab_verification_t4v.py --load "../../${model_pth}" --model cnn_model0 --batch_size 400 --timeout 3600  --eps $eps --img $image_num >> "../../${out_log_pth}"
    >&2 echo "Processed image $image_num"
    ((image_num++))

    echo "" >> "../../${out_log_pth}"
done