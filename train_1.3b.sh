# pip install lightning
NUM_NODES=8
torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT examples/wanvideo/train_wan_1.3b_motioncanvas.py \
    --task train \
    --train_architecture full \
    --dataset_path <DATA_PATH> \
    --output_path <OUTPUT_PATH>  \
    --dit_path <DIT_PATH> \
    --vae_path <VAE_PATH> \
    --text_encoder_path <TEXT_ENCODER_PATH> \
    --image_encoder_path <IMAGE_ENCODER_PATH> \
    --invalid_data_path <TXT_PATH> \
    --moving_noun_path .data/filtered_ram_tag_list.txt \
    --resume_from <RESUME_CKPT_PATH> \
    --max_epochs 10000 \
    --learning_rate 5e-5 \
    --accumulate_grad_batches 1 \
    --use_gradient_checkpointing \
    --batch_size 1 \
    --dataloader_num_workers 8 \
    --every_n_train_steps 1000 \
    --num_frames 49 \
    --training_strategy deepspeed_stage_1 \
    --num_nodes $NUM_NODES \
    --target_fps 15

