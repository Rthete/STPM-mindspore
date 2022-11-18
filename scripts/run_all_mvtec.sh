#!/bin/bash

if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_all_mvtec.sh DATASET_PATH BACKONE_PATH DEVICE_NUM"
    echo "For example: bash run_all_mvtec.sh /path/dataset /path/backbone_ckpt 1"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

script_path=$(cd "$(dirname $0)" || exit; pwd)

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
CKPT_APTH=$(get_real_path $2)
device_num=$3

total_num=15
sclie=`echo "scale=2; $total_num/$device_num" | bc`
sclie=$(printf "%.f\n" "$sclie")
mod=$(($total_num % $device_num))

arr=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

exec_train_eavl(){
  for value in $arr_sub
  do
    sleep 2s
    bash run_standalone_train.sh $DATA_PATH $CKPT_APTH $value $device_id
    eval_ckpt=$script_path/train_${value}/ckpt/${value}_best_added.ckpt
    bash run_eval.sh  $DATA_PATH $eval_ckpt $value $device_id
  done
}

for((i=0;i<$device_num;i++));
do
  start=$(($i * $sclie))
  end=$((($i+1) * $sclie))
  if [ $i == $(($device_num -1)) ] && [ $device_num != 1 ]; then
    sclie=$(($sclie + $mod))
    end=$total_num
  fi
  arr_sub=${arr[*]:$start:$sclie}
  device_id=$i
  echo "start: $start, end: $end, category: "$arr_sub, $device_id
  exec_train_eavl $arr_sub &> log$i &
done
