python -m torch.distributed.launch --master_port 25901 --nproc_per_node 8 \
       ./main.py {dataset-path} \
       --model ViTAEv2_VSA_S \
       -b 128 --lr 5e-4 --weight-decay .05 --img-size 224 --workers 8
