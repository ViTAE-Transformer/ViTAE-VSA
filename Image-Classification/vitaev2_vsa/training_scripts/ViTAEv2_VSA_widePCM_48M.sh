python -m torch.distributed.launch \
       --nnodes 2 --node_rank {} --master_addr {} --master_port 25900 --nproc_per_node 8 \
       ./main.py {dataset-path} \
       --model ViTAEv2_VSA_widePCM_48M \
       -b 64 --lr 5e-4 --weight-decay .055 --img-size 224 --drop-path 0.25 --workers 8 \
