. /home/fangjr/Envs/horovod/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
# mpirun -np 4 python3 ./cifar10_main.py --batch_size=32 --resnet_size=50
export NUM_NODES=4
export BATCH_SIZE=2048
export RESNET_SIZE=20

for BATCH_SIZE in 512 1024 2048; do
  export LOG_DIR=./log/resnet${RESNET_SIZE}_${NUM_NODES}_node_${BATCH_SIZE}_batch_warmup
  #export LOG_DIR=./archive/resnet50-warmup/resnet50_4_node_2048_batch_warmup/
#for BATCH_SIZE in 64; do
  mpirun -np ${NUM_NODES} python3 ./cifar10_main.py \
    --batch_size=${BATCH_SIZE} \
    --resnet_size=${RESNET_SIZE} \
    --data_dir=/home/fangjr/dataset \
    --train_epochs=200 \
    --model_dir=$LOG_DIR \
    2>&1 $LOG_DIR/run.log
done;
deactivate
