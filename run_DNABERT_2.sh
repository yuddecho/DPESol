CURR_PATH="$(pwd)"
echo "run.sh CURR_PATH: ${CURR_PATH}"
export PATH="${CURR_PATH}:${PATH}"

PARENT_PATH=$(dirname "$PWD")
echo "run.sh PARENT_PATH: ${PARENT_PATH}"
export PYTHONPATH="${PARENT_PATH}:${PYTHONPATH}"

#NUM_GPUS=4
#python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} training.py

#python ./model/train.py --web_mode
#python ./model/train.py
python ./model/DNABERT_2_Embedding.py

