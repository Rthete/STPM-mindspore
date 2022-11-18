#!/bin/bash
if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec.sh DATASET_PATH CKPT_APTH CATEGORY DEVICE_ID"
    echo "For example: bash run_eval.sh /path/dataset /path/ckpt category 1"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
CKPT_APTH=$(get_real_path $2)
category=$3
device_id=$4

eval_path=eval_$category
if [ -d $eval_path ];
then
    rm -rf ./$eval_path
fi
mkdir ./$eval_path
cd ./$eval_path
env > env0.log
echo "[INFO] start eval dataset $category with device_id: $device_id"
python ../../eval.py \
--data_url $DATA_PATH  \
--ckpt_path $CKPT_APTH \
--category $category \
--device_id $device_id \
> eval.log 2>&1

if [ $? -eq 0 ];then
    echo "[INFO] eval success"
else
    echo "[ERROR] eval failed"
    exit 2
fi
echo "[INFO] finish"
cd ../
