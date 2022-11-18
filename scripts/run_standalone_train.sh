#!/bin/bash
if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec.sh DATASET_PATH BACKONE_PATH CATEGORY DEVICE_ID"
    echo "For example: bash run_standalone_train.sh /path/dataset /path/backbone_ckpt category 1"
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

train_path=train_$category
if [ -d $train_path ];
then
    rm -rf ./$train_path
fi
mkdir ./$train_path
cd ./$train_path
env > env0.log
echo "[INFO] start train dataset $category with device_id: $device_id"
python ../../train.py \
--data_url $DATA_PATH \
--pre_ckpt_path $CKPT_APTH \
--category $category \
--device_id $device_id \
> train.log 2>&1

if [ $? -eq 0 ];then
    echo "[INFO] training success"
else
    echo "[ERROR] training failed"
    exit 2
fi
echo "[INFO] finish"
cd ../
