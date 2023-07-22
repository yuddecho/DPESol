CURR_PATH="$(pwd)"
export PATH="${CURR_PATH}:${PATH}"

parent_dir=$(dirname "$PWD")
export PYTHONPATH="${parent_dir}:${PYTHONPATH}"

#NUM_GPUS=4
#python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} training.py

python ./model/train.py --web_mode