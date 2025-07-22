while true #dynamic choose port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT


torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=$PORT \
    tools/train.py \
    --base configs/training.yaml \
    --finetune ckpts/vista.safetensors \
    --num_nodes 1 \
    --n_devices 8 
