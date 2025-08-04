import os
import sys
diffsynth_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(diffsynth_dir)

import torch, argparse
import lightning as pl
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffsynth import WanVideoPipeline_motioncanvas, ModelManager, save_video
from diffsynth.pipelines.tracker_utils import get_video_track_video
from dataset import TextVideoDataset_oss, collate_fn
from logger_utils import setup_logger
from vis_utils import draw_annotations, video_stitching
from format_utils import tensor_to_video_list
from model_utils import load_checkpoints
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path, vae_path, dit_path,
        image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        p_drop_bbox=0.1, p_drop_traj=0.1, visualize_step=1000,
        logger_=None,
    ):
        super().__init__()
        self.logger_ = logger_
        model_path = [text_encoder_path, vae_path]
        # model_path = [vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline_motioncanvas.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.pipe.bbox_zeroconv.requires_grad_(True)

        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # self.vae_channel = 16
        # self.downsample_ratios = [4, 8, 8]
        torch.hub.set_dir('/mnt/29a30e4acb8/jinboxing/cache/torch_cache/hub')
        self.cotracker = torch.hub.load("/mnt/29a30e4acb8/jinboxing/cache/torch_cache/hub/facebookresearch_co-tracker_main", "cotracker3_offline", source="local").to('cuda', dtype=torch.bfloat16)
        self.cotracker.requires_grad_(False)

        self.p_drop_bbox=p_drop_bbox
        self.p_drop_traj=p_drop_traj
        self.visualize_step=visualize_step
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe.bbox_zeroconv.train()
        
    
    def training_step(self, batch, batch_idx):
        # Data
        text, video, video_rgb, bbox_mask, reference_imgs_indicator = batch["text"], batch["video"], batch["video_rgb"], batch["bbox_mask"], batch["reference_imgs_indicator"]
        # 
        object_masks, object_bbox_masks = batch["object_masks"], batch["object_bbox_masks"]  # [num_objects, T, H, W]
        #
        bbox_info = batch["bbox_info"]  # List of bbox info for each frame
        # self.logger_.info(f'###### BBOX: {bbox_info}')
        self.pipe.device = self.device
        # self.logger_.info("##### Start VAE and text encode !!")
        with torch.no_grad():
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)  #  torch.Size([1, 512, 4096])
            # logger.info("##### Text encode finished !!")
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            video_rgb = video_rgb.to(dtype=torch.float32, device=self.device)
            bbox_mask = bbox_mask.to(dtype=self.pipe.torch_dtype, device=self.device)

            latents = self.pipe.encode_video(video, **self.tiler_kwargs)
            p_drop_bbox = torch.rand(1).item()
            if p_drop_bbox < self.p_drop_bbox:
                bbox_mask = torch.zeros_like(bbox_mask).to(dtype=self.pipe.torch_dtype, device=self.device)

        # self.logger_.info("##### VAE encode finished !!")
        if "first_frame" in batch:
            first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
            _, _, num_frames, height, width = video.shape
            image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
        else:
            image_emb = {}

        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        
        # self.logger_.info(f"##### image_emb: {image_emb['y'].shape} {image_emb['clip_feature'].shape}")
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"].to(self.device)

        vid_f, vid_h, vid_w = video.shape[2], video.shape[3], video.shape[4]

        # self.logger_.info('###### preparing kwargs')
        bbox_latents, track_video, track_info = self.pipe.prepare_motioncanvas_kwargs(
            video_rgb=video_rgb,
            video_frame_num=vid_f,
            bbox_mask=bbox_mask,
            reference_imgs_indicator=reference_imgs_indicator,
            object_bbox_masks=object_bbox_masks,
            object_masks=object_masks,
            cotracker=self.cotracker,
            tiler_kwargs=self.tiler_kwargs
            )
        # self.logger_.info(f'##### Track info: {track_info.shape} {track_info}')
        if torch.rand(1).item() < self.p_drop_traj:
            track_video = torch.zeros_like(track_video).to(latents.dtype)

        # self.logger_.info('###### computing loss')
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noisy_latents = noisy_latents + bbox_latents
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, track_video, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        # logger.info("##### Model Forward finished !!")
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        # self.logger_.info("##### Loss Compute finished !!")

        # visualization during training
        if self.global_step % self.visualize_step == 0:
            self.visualize_during_training(latents.shape, text, first_frame, reference_imgs_indicator, bbox_mask, track_video, object_bbox_masks, object_masks, bbox_info, track_info, video_rgb)
            self.pipe.scheduler.set_timesteps(1000, training=True)

        return loss


    def configure_optimizers(self):
        denoising_trainable_params = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        zeroconv_trainable_params = filter(lambda p: p.requires_grad, self.pipe.bbox_zeroconv.parameters())
                       
        trainable_modules = list(denoising_trainable_params) + list(zeroconv_trainable_params) 

        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.02, total_iters=500)

        # print trainable parameters
        for key, param in self.pipe.denoising_model().named_parameters():
            if param.requires_grad:
                logger.info(f'Train DiT: {key}')
        for key, param in self.pipe.bbox_zeroconv.named_parameters():
            if param.requires_grad:
                logger.info(f'Train bbox_zeroconv: {key}')        
        return [optimizer], [scheduler]
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        learnable_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                full_name = prefix + name
                if destination is None:
                    learnable_dict[full_name] = param if keep_vars else param.detach()
                else:
                    destination[full_name] = param if keep_vars else param.detach()
        
        return learnable_dict if destination is None else destination
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            elif strict:
                raise RuntimeError(f"Unexpected key {name} in state_dict")
        
        # handle missing keys
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if missing:
                raise RuntimeError(f"Missing keys in state_dict: {missing}")
    
    def visualize_during_training(self, latents_shape, text, first_frame, reference_imgs_indicator, bbox_mask, track_video, object_bbox_masks, object_masks, bbox_info, track_info, video_rgb=None):
        generated_frames_list = self.pipe(
            prompt=text,
            input_image=first_frame,
            negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            num_inference_steps=50,
            tiled=True, 
            # visualization parameters
            training_visualization=True,
            latents_shape=latents_shape,
            reference_imgs_indicator=reference_imgs_indicator,
            bbox_mask=bbox_mask,
            track_video=track_video,
            object_bbox_masks=object_bbox_masks, 
            object_masks=object_masks,
        )
        # generated_frames_list: List of PIL Image [0, 255] uint8
        generated_frames_list = draw_annotations(generated_frames_list, bbox_info, track_info)
        if video_rgb is not None:
            gt_frames_list = tensor_to_video_list(video_rgb)
            gt_frames_list = draw_annotations(gt_frames_list, bbox_info, track_info)
            generated_frames_list = video_stitching(gt_frames_list, generated_frames_list)
        save_dir = os.path.join(self.trainer.default_root_dir, "training_visualizations")
        os.makedirs(save_dir, exist_ok=True)
        for vid, video in enumerate(generated_frames_list):
            save_path = os.path.join(save_dir, f"step{self.global_step}_rank{dist.get_rank()}_video{vid}.mp4")  
            save_video(video, save_path, fps=15, quality=5)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        # default=81,
        default=49,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size.",
    )
    parser.add_argument(
        "--invalid_data_path",
        type=str,
        default=None,
        help="Path to the invalid data file.",
    )
    parser.add_argument(
        "--moving_noun_path",
        type=str,
        default=None,
        help="Path to the moving noun file.",
    )
    parser.add_argument("--target_fps", type=int, default=15, help="target fps",)
    parser.add_argument("--every_n_train_steps", type=int, default=100, help="every n train steps",)
    parser.add_argument("--visualize_step", type=int, default=1000, help="visualize_step",)
    parser.add_argument("--num_nodes", type=int, default=1, help="the number of nodes for training",)
    parser.add_argument("--resume_from", type=str, default="", help="the ckpt path to resume",)
    args = parser.parse_args()
    return args

def train(args, logger):
    dataset = TextVideoDataset_oss(
        mask_path=os.path.join(args.dataset_path, "data_debug.csv") if args.debug else os.path.join(args.dataset_path, "data.csv"),
        target_fps=args.target_fps,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        logger=logger,
        invalid_data_path=args.invalid_data_path,
        moving_noun_path=args.moving_noun_path,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=None if args.dataloader_num_workers == 0 else 2,
        collate_fn=collate_fn,
    )
    model = LightningModelForTrain(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        visualize_step=args.visualize_step,
        logger_=logger,
    )
    resume_checkpoint_path = args.resume_from if args.resume_from != "" else os.path.join(args.output_path, "resume_checkpoints", "last.ckpt", 'checkpoint', 'mp_rank_00_model_states.pt')
    # model = model.load_from_checkpoint(checkpoint_path=resume_checkpoint_path) if os.path.exists(resume_checkpoint_path) else model
    if os.path.exists(resume_checkpoint_path):
        logger.info(f">>> Resuming from checkpoint: {resume_checkpoint_path}")
        model = load_checkpoints(model, torch.load(resume_checkpoint_path, map_location="cpu"), logger)

    # Initialize TensorBoardLogger
    tb_logger = TensorBoardLogger(
        save_dir=args.output_path, 
        name="tensorboard_logs"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_path, "resume_checkpoints"),
        filename="ckpt-{step}",
        every_n_train_steps=args.every_n_train_steps,
        save_weights_only=True,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )


    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=1,
    )
    # trainer.fit(model, dataloader, ckpt_path=resume_checkpoint_path if os.path.exists(resume_checkpoint_path) else None)
    logger.info(f">>> Starting training, output path: {args.output_path}")
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    logger = setup_logger(args.output_path)
    
    if args.task == "train":
        train(args, logger)