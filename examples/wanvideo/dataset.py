import os
import torch
import torchvision
import ossutils as ops
import decord
decord.bridge.set_bridge('torch')
import numpy as np
import random
import itertools
import pandas as pd

from PIL import Image
from io import BytesIO
from cond_utils import MaskJsonLoader
from pathlib import Path
from torchvision.transforms import v2
from einops import rearrange
from logger_utils import setup_logger
from tqdm import tqdm
from diffsynth import save_video

def collate_fn(batch):
    collated_batch = {}
    collated_batch['text'] = [item['text'] for item in batch]
    # video: [B, C, T, H, W]）
    collated_batch['video'] = torch.stack([item['video'] for item in batch], dim=0)
    # RGB video: [B, T, C, H, W]）
    collated_batch['video_rgb'] = torch.stack([item['video_rgb'] for item in batch], dim=0)
    # Bbox: [B, C, T, H, W]）
    collated_batch['bbox_mask'] = torch.stack([item['bbox_mask'] for item in batch], dim=0)

    # first_frame: [B, 3, H, W] (only for I2V task)
    if 'first_frame' in batch[0]:
        collated_batch['first_frame'] = torch.stack([torch.from_numpy(item['first_frame']) for item in batch], dim=0)  # [B, 3, H, W]
    # object_masks:
    obj_masks_list = [item['object_masks'] for item in batch]  # List of [n_i, T, 1, H, W]
    collated_batch['object_masks'] = torch.cat(obj_masks_list, dim=0)
    # object_bbox_masks:
    obj_bbox_masks_list = [item['object_bbox_masks'] for item in batch]  # List of [n_i, T, 1, H, W]
    collated_batch['object_bbox_masks'] = torch.cat(obj_bbox_masks_list, dim=0)

    # calculate bbox number
    collated_batch['reference_imgs_indicator'] = [item['object_bbox_masks'].shape[0] for item in batch]

    collated_batch['bbox_info'] = [item['bbox_info'] for item in batch]
    return collated_batch

def adjust_bbox_to_center_crop(
    bboxes, 
    original_size,  # (H, W)
    target_size,    # (target_h, target_w) 
    center_crop_size,  # (crop_h, crop_w)
    resize_mode="max"  # 
):
    """
    Adjust bbox after:
        1. crop_and_resize 
        2. CenterCrop
    
    Args:
        bboxes (list): [x1, y1, x2, y2] 
        original_size (tuple): (H, W)
        target_size (tuple): (target_h, target_w) 
        center_crop_size (tuple): (crop_h, crop_w)
        resize_mode (str): "max" 
    
    Returns:
        list: [new_x1, new_y1, new_x2, new_y2] 
    """
    H, W = original_size
    target_h, target_w = target_size
    crop_h, crop_w = center_crop_size
    

    scale = max(target_w / W, target_h / H)

    scaled_h = round(H * scale)
    scaled_w = round(W * scale)
    
    crop_x1 = max(0, (scaled_w - crop_w) // 2)
    crop_y1 = max(0, (scaled_h - crop_h) // 2)
    
    x1, y1, x2, y2 = bboxes
    
    scaled_x1 = x1 * scale
    scaled_y1 = y1 * scale
    scaled_x2 = x2 * scale
    scaled_y2 = y2 * scale
    
    new_x1 = scaled_x1 - crop_x1
    new_y1 = scaled_y1 - crop_y1
    new_x2 = scaled_x2 - crop_x1
    new_y2 = scaled_y2 - crop_y1
    
    new_x1 = max(0, min(new_x1, crop_w))
    new_y1 = max(0, min(new_y1, crop_h))
    new_x2 = max(0, min(new_x2, crop_w))
    new_y2 = max(0, min(new_y2, crop_h))
    
    return [new_x1, new_y1, new_x2, new_y2]


class TextVideoDataset_oss(torch.utils.data.Dataset):
    def __init__(self, mask_path, logger=None, target_fps=15, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, use_all_frames=False, preprocess_data=False, invalid_data_path=None, moving_noun_path=None):
        self.logger = logger
        mask_data = pd.read_csv(mask_path)
        mask_data = mask_data.sample(frac=1).reset_index(drop=True)
        self.dataset_len = len(mask_data)
        self.mask_video_oss_paths = mask_data["oss_key"].to_list()
        self.text = mask_data["caption"].to_list()
        self.mask_cond_oss_paths = mask_data["external_info"].to_list()

        self.oss_video_dir = os.path.join(ops.vgd_bucket)
        self.oss_cond_dir = os.path.join(ops.vgd_bucket)

        self.mask_json_loader = MaskJsonLoader()
        assert invalid_data_path is not None
        self.invalid_data_path = invalid_data_path
        if not os.path.exists(invalid_data_path):
            Path(self.invalid_data_path).touch()
        with open(self.invalid_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.invalid_data = f.readlines()

        assert moving_noun_path is not None
        with open(moving_noun_path, 'r') as f:
            self.moving_noun = [line.strip() for line in f.readlines()]
            
        self.target_fps = target_fps
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.use_all_frames = use_all_frames
        self.preprocess_data = preprocess_data
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # CenterCrop + Resize）
        self.raw_frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
        ])

        self.ref_frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
        ])

        self.obj_mask_frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), interpolation=v2.InterpolationMode.NEAREST),
            v2.ToTensor(),
        ])

        # color_palette for color-coded bbox
        self.color_palette = [
            [64, 0, 0],
            [0, 64, 0],
            [0, 0, 64],
            [191, 0, 0],
            [0, 191, 0],
            [0, 0, 191],
        ]
        
        
    def crop_and_resize(self, image, interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=interpolation
        )
        return image


    def load_frames_using_decord(self, file_path, target_fps, num_frames, frame_process):
        try:
            video_bytes = ops.read(file_path, mode='rb')
        except Exception as e:
            # logger.info(f'get_object from {file_path} failed with error {e}', flush=True)
            return None
        reader = decord.VideoReader(BytesIO(video_bytes))
    
        max_num_frames = len(reader)
        original_fps = reader.get_avg_fps()

        if self.use_all_frames or self.preprocess_data:
            frame_indices = list(range(max_num_frames))
        else:
            frames_by_interval2 = (num_frames - 1) * 2 + 1
            if max_num_frames >= frames_by_interval2:
                interval = max(1, round(original_fps / target_fps))
            else:
                interval = 1
            total_frames_needed = (num_frames - 1) * interval + 1
            if max_num_frames < total_frames_needed:
                return None
        
            max_start = max_num_frames - total_frames_needed
            if max_start <= 0:
                start_frame_id = 0
            else:
                start_frame_id = torch.randint(0, max_start + 1, (1,)).item()
            
            frame_indices = [start_frame_id + i * interval for i in range(num_frames)]
        
        if not self.preprocess_data:
            if max_num_frames > len(frame_indices):
                all_frames = set(range(max_num_frames))
                sampled_set = set(frame_indices)
                available_frames = list(all_frames - sampled_set)
                ref_frame_idx = np.random.choice(available_frames)
            else:
                ref_frame_idx = np.random.randint(0, max_num_frames)
            
            ref_frame_tensor = reader[ref_frame_idx]
            ref_frame_np = ref_frame_tensor.numpy()
            ref_frame = Image.fromarray(ref_frame_np)
            ref_frame = self.crop_and_resize(ref_frame)
            ref_frame = self.ref_frame_process(ref_frame)
            ref_frame = np.array(ref_frame)
            
        video_frames = reader.get_batch(frame_indices)
        
        frames = []
        rgb_frames = []
        first_frame = None
        for i in range(num_frames):
            frame_tensor = video_frames[i]
            frame_np = frame_tensor.numpy()
            frame = Image.fromarray(frame_np)
            frame = self.crop_and_resize(frame)
            if self.is_i2v and i == 0:
                first_frame = frame.copy()
            raw_frame = self.raw_frame_process(frame) * 255.0
            raw_frame = raw_frame.to(torch.uint8)
            rgb_frames.append(raw_frame)
            frame = frame_process(frame)
            frames.append(frame)
        
        del reader
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        rgb_frames = torch.stack(rgb_frames, dim=0)  # [T, C, H, W]
        if first_frame is not None:
            first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
            first_frame = np.array(first_frame)
        
        if self.is_i2v:
            return frames, rgb_frames, first_frame, frame_indices
        else:
            return frames, rgb_frames, frame_indices


    def load_video(self, file_path, mask_array, valid_ids):
        return self.load_frames_using_decord(file_path, self.target_fps, self.num_frames, self.frame_process)
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __len__(self):
        return len(self.mask_video_oss_paths)
    
    def __getitem__(self, data_id):
        # logger.info(f"Processing data_id: {data_id}")
        while data_id < self.dataset_len:  # Loop until we find a valid item
            try:
                text = self.text[data_id]
                mask_video_oss_path = self.mask_video_oss_paths[data_id]
                mask_cond_oss_path = self.mask_cond_oss_paths[data_id].replace('oss://sora-data/', '')

                if mask_video_oss_path in self.invalid_data:
                    # logger.info(f"Skipping invalid data: {mask_video_oss_path}")
                    data_id = (data_id + 1) % self.dataset_len  # Ensure data_id wraps around
                    continue

                mask_video_oss_full_path = os.path.join(self.oss_video_dir, mask_video_oss_path)
                mask_cond_oss_full_path = os.path.join(self.oss_cond_dir, mask_cond_oss_path)

                ori_mask_info_dict = self.mask_json_loader(mask_cond_oss_full_path)  
                # logger.info(f"Loaded mask JSON: {mask_cond_oss_path}")
                valid_ids, color_mask, bbox_mask, bbox_info, obj_masks, obj_bbox_masks = self.process_mask(ori_mask_info_dict) 
                # self.logger.info(f"Valid IDs: {valid_ids}, Color Mask Shape: {color_mask.shape}, BBox Mask Shape: {bbox_mask.shape}, Object Masks Shape: {obj_masks.shape}, Object BBox Masks Shape: {obj_bbox_masks.shape}")

                # logger.info(f"Processed mask data | Objects: {len(valid_ids)}")
                if len(valid_ids) > 0 and len(bbox_info) > 0 and len(bbox_info[0]) > 0:
                    
                    video_result = self.load_video(mask_video_oss_full_path, ori_mask_info_dict[1], valid_ids)

                    if video_result is None:
                        # self.logger.info(f"Video loading failed: {mask_video_oss_path}")
                        with open(self.invalid_data_path, 'a') as invalid_data:
                            invalid_data.write(f"{mask_video_oss_path}\n")
                        data_id = (data_id + 1) % self.dataset_len  # Ensure data_id wraps around
                        continue
                    
                    # self.logger.info(f"Loaded video: {mask_video_oss_path}")
                    
                    if self.is_i2v:
                        video, video_rgb, first_frame, frame_indices = video_result
                    else:
                        video, video_rgb, frame_indices = video_result
                    
                    sampled_bbox_mask = bbox_mask[frame_indices]
                    sampled_obj_masks = obj_masks[:, frame_indices]
                    sampled_obj_bbox_masks = obj_bbox_masks[:, frame_indices]

                    sampled_bbox_mask_frames = []
                    sampled_obj_mask_frames = []
                    sampled_obj_bbox_mask_frames = []

                    bbox_info_processed = [[] for _ in range(sampled_bbox_mask.shape[0])]
                    # Iterate over time sequence and objects
                    obj_num = sampled_obj_masks.shape[0]
                    for i in range(sampled_bbox_mask.shape[0]):  # assuming len(sampled_bbox_mask) is the time dimension
                        bbox_frame_np = sampled_bbox_mask[i]
                        bbox_frame = Image.fromarray(bbox_frame_np)
                        bbox_frame = self.crop_and_resize(bbox_frame)
                        bbox_frame = self.frame_process(bbox_frame)
                        sampled_bbox_mask_frames.append(bbox_frame)

                        obj_per_frame = []
                        obj_bbox_per_frame = []
                        for obj_index in range(obj_num):
                            obj_mask_np = sampled_obj_masks[obj_index, i]  # Get mask for specific object at time i
                            obj_bbox_mask_np = sampled_obj_bbox_masks[obj_index, i]
                            # Assume single-channel (grayscale) for binary mask
                            obj_mask_frame = Image.fromarray(obj_mask_np)
                            obj_bbox_mask_frame = Image.fromarray(obj_bbox_mask_np)

                            obj_mask_frame = self.crop_and_resize(obj_mask_frame, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                            obj_mask_frame = self.obj_mask_frame_process(obj_mask_frame)
                            obj_mask_frame = obj_mask_frame > 0
                            obj_per_frame.append(obj_mask_frame)

                            obj_bbox_mask_frame = self.crop_and_resize(obj_bbox_mask_frame, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                            obj_bbox_mask_frame = self.obj_mask_frame_process(obj_bbox_mask_frame)
                            obj_bbox_mask_frame = obj_bbox_mask_frame > 0
                            obj_bbox_per_frame.append(obj_bbox_mask_frame)

                            # self.logger.info(bbox_info)
                            bbox_info_processed[i].append({
                                'bbox_coord': adjust_bbox_to_center_crop(bbox_info[i][obj_index]['bbox'], (bbox_info[i][obj_index]['orig_size'][1], bbox_info[i][obj_index]['orig_size'][0]), (self.height, self.width), (self.height, self.width)),
                            })
                        
                        sampled_obj_mask_frames.append(torch.stack(obj_per_frame, dim=0))
                        sampled_obj_bbox_mask_frames.append(torch.stack(obj_bbox_per_frame, dim=0))

                    sampled_bbox_mask_frames = torch.stack(sampled_bbox_mask_frames, dim=0)
                    sampled_bbox_mask_frames = rearrange(sampled_bbox_mask_frames, "T C H W -> C T H W")
                    sampled_obj_mask_frames = torch.stack(sampled_obj_mask_frames, dim=0)
                    sampled_obj_mask_frames = rearrange(sampled_obj_mask_frames, "T N C H W -> N T C H W")
                    sampled_obj_bbox_mask_frames = torch.stack(sampled_obj_bbox_mask_frames, dim=0)
                    sampled_obj_bbox_mask_frames = rearrange(sampled_obj_bbox_mask_frames, "T N C H W -> N T C H W")

                    # self.logger.info(f"Frame processing completed")

                    data = {
                        "text": text, 
                        "video": video,  # [3, 49, 480, 832]  [C, T, H, W]
                        "video_rgb": video_rgb, # [49, 3, 480, 832]  [T, C, H, W]
                        "bbox_mask": sampled_bbox_mask_frames,  # [3, 49, 480, 832]  [C, T, H, W]
                        "object_masks": sampled_obj_mask_frames,  # [3, 49, 1, 480, 832] [obj_num, T, 1, H, W]
                        "object_bbox_masks": sampled_obj_bbox_mask_frames,  # [3, 49, 1, 480, 832] [obj_num, T, 1, H, W]
                        "bbox_info": bbox_info_processed,
                    }

                    if self.is_i2v:
                        data["first_frame"] = first_frame
                    
                    return data
                else:
                    # record invalid data if mask is invalid
                    # self.logger.info(f"Invalid mask data: {mask_video_oss_path}")
                    with open(self.invalid_data_path, 'a') as invalid_data:
                        invalid_data.write(f"{mask_video_oss_path}\n")
                    data_id = (data_id + 1) % self.dataset_len  # Ensure data_id wraps around
            
            except Exception as e:
                # self.logger.info(e)
                with open(self.invalid_data_path, 'a') as invalid_data:
                    invalid_data.write(f"{mask_video_oss_path}\n")
                data_id = (data_id + 1) % self.dataset_len  # Ensure data_id wraps around
                continue

        raise IndexError("No valid data found within the dataset range.")

    def process_mask(self, mask):
        '''
            mask is a tuple (id_to_names, mask_dict)
            id_to_names: dict{
                "<obj_id>": "<obj_class_name>"
            }
            mask_dict : dict {
                "<obj_id>": {
                    'masks': np.array,  # [T, H, W]
                    'areas': np.array,  # [T]
                    'boxes': np.array   # [T, 4] (xmin, ymin, xmax, ymax)
                },
            }
        '''
        id_and_names = mask[0]
        mask_info_dict = mask[1]
        T, H, W = next(iter(mask_info_dict.values()))['masks'].shape
        
        # initialize mask as black
        color_mask = np.zeros((T, H, W, 3), dtype=np.uint8) * 255
        bbox_mask = np.zeros((T, H, W, 3), dtype=np.uint8) * 255
        
        
        valid_ids = [id for id, name in id_and_names.items() if name in self.moving_noun]
        # self.logger.info(f'valid_ids: {valid_ids}')
        sampled_colors = random.sample(self.color_palette, min(len(valid_ids), len(self.color_palette)))
        color_cycle = itertools.cycle(sampled_colors)
        id_to_color = {valid_id: next(color_cycle) for valid_id in valid_ids}
        
        # initialize masks and bbox masks
        obj_masks = np.zeros((len(valid_ids), T, H, W), dtype=np.uint8)
        obj_bbox_masks = np.zeros((len(valid_ids), T, H, W), dtype=np.uint8)

        bbox_info = [[] for _ in range(T)]  # A list for each frame to store bbox info

        # logger.info("------Process Mask Init Finished!!")
        
        # area info to detect bbox area change
        prev_areas = {obj_id: [None] * T for obj_id in valid_ids}
        for obj_id in valid_ids:
            obj_info = mask_info_dict[obj_id]
            areas = obj_info['areas']  
            for t in range(T):
                prev_areas[obj_id][t] = areas[t]
        
        # logger.info("------Process Mask First Loop Finished!!")
        
        for idx, obj_id in enumerate(valid_ids):
            obj_info = mask_info_dict[obj_id]
            binary_mask = obj_info['masks'] 
            boxes = obj_info['boxes']  
            areas = obj_info['areas'] 
            color = np.array(id_to_color[obj_id], dtype=np.uint8) 
            obj_name = id_and_names[obj_id]
            
            for t in range(T):
                frame_mask = binary_mask[t]
                area = areas[t]
                box = boxes[t]  # [x1, y1, x2, y2]
                
                # detect area sudden change (wrt. the previous frame)
                is_valid = True
                if t > 0:
                    prev_area = prev_areas[obj_id][t-1]
                    if prev_area is not None and area is not None:
                        area_change = abs(area - prev_area) / (prev_area + 1e-5) 
                        if area_change > 0.5:
                            is_valid = False
                
                if is_valid and area > 0: 
                    color_mask[t][frame_mask == 1] = color
                    
                    xmin, ymin, xmax, ymax = box
                    xmin = max(0, min(xmin, W - 1))
                    ymin = max(0, min(ymin, H - 1))
                    xmax = max(0, min(xmax, W - 1))
                    ymax = max(0, min(ymax, H - 1))
                    
                    bbox_mask[t, ymin:ymax, xmin:xmax] += color

                    obj_masks[idx, t] = frame_mask.astype(np.uint8)
                    obj_bbox_masks[idx, t, ymin:ymax, xmin:xmax] = 1
                    
                    bbox_info[t].append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'name': obj_name,
                        'color': color,
                        'orig_size': (W, H)
                    })

        # valid_ids: [], color_mask: [T,H,W,C], bbox_mask: [T,H,W,C], obj_masks: [obj_num,T,H,W], obj_bbox_masks: [obj_num,T,H,W]
        # self.logger.info(f"Processed mask data | Objects: {len(valid_ids)}, Color Mask Shape: {color_mask.shape}, BBox Mask Shape: {bbox_mask.shape}, Object Masks Shape: {obj_masks.shape}, Object BBox Masks Shape: {obj_bbox_masks.shape}")
        return valid_ids, color_mask, bbox_mask, bbox_info, obj_masks, obj_bbox_masks
    
def overlay_bbox_on_video(video_tensor, bbox_tensor, save_path, fps=25):
    """
    Overlay BBox mask onto video and save as MP4 file
    
    Args:
        video_tensor: Video tensor with shape [1, 3, T, H, W]
        bbox_tensor: BBox mask tensor with shape [1, 3, T, H, W]
        save_path: Path to save the output video
        fps: Frame rate of the output video
    """
    # Remove batch dimension and convert to numpy array
    video = video_tensor.squeeze(0).permute(1, 2, 3, 0).numpy()  # [T, H, W, 3]
    bbox = bbox_tensor.squeeze(0).permute(1, 2, 3, 0).numpy()   # [T, H, W, 3]
    
    # Convert values from [-1, 1] range to [0, 1] range
    video = (video + 1) / 2
    bbox = (bbox + 1) / 2
    
    # Create list of overlayed frames
    frames = []
    for v_frame, b_frame in tqdm(zip(video, bbox), total=len(video), desc="Processing frames"):
        # Overlay BBox mask (multiplied by 0.5) onto video frame
        overlay = v_frame + b_frame * 0.5
        # Clip to [0, 1] range
        overlay = np.clip(overlay, 0, 1)
        # Convert to [0, 255] range and change data type
        overlay = (overlay * 255).astype(np.uint8)
        frames.append(overlay)
    
    # Save video
    save_video(frames, save_path, fps)

if __name__ == "__main__":
    def log_tensor_stats(name, tensor):
        if tensor is not None:
            logger.info(f"{name}:")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Dtype: {tensor.dtype}")
            logger.info(f"  Min: {tensor.min().item():.4f}")
            logger.info(f"  Max: {tensor.max().item():.4f}")
        else:
            logger.info(f"{name}: None")
    mask_path = '/mnt/29a30e4acb8/jinboxing/Data/VideoAnnotation/data.csv'
    logger = setup_logger('/mnt/29a30e4acb8/jinboxing/Experiments/DiffSynth-MotionCanvas/exp01/')
    dataset = TextVideoDataset_oss(
        mask_path=mask_path,
        logger=logger,
        target_fps=15,
        frame_interval=1,
        num_frames=49, 
        height=480,
        width=832,
        is_i2v=True,
        use_all_frames=False,
        preprocess_data=False,
        invalid_data_path='/mnt/29a30e4acb8/jinboxing/Data/VideoAnnotation/invalid_data.txt',
        moving_noun_path='/mnt/29a30e4acb8/jinboxing/Data/VideoAnnotation/filtered_ram_tag_list.txt'
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i}:")
        logger.info(f"Text: {batch['text']}")
        log_tensor_stats("Video", batch.get('video'))
        log_tensor_stats("Video RGB", batch.get('video_rgb'))
        log_tensor_stats("BBox mask", batch.get('bbox_mask'))
        log_tensor_stats("Object masks", batch.get('object_masks'))
        log_tensor_stats("Object bbox masks", batch.get('object_bbox_masks'))
        log_tensor_stats("First frame", batch.get('first_frame'))
        logger.info(f"Object bbox info: {batch.get('bbox_info')}")
        overlay_bbox_on_video(batch.get('video'), batch.get('bbox_mask'), f"/mnt/29a30e4acb8/jinboxing/Experiments/DiffSynth-MotionCanvas/exp01/{i}.mp4", fps=15)
        logger.info("-" * 20)

'''
[2025-07-15 23:10:23][Rank 0][INFO] Text: ['a test caption']
[2025-07-15 23:10:23][Rank 0][INFO] Video shape: torch.Size([1, 3, 49, 480, 832])
[2025-07-15 23:10:23][Rank 0][INFO] Video RGB shape: torch.Size([1, 49, 3, 480, 832])
[2025-07-15 23:10:23][Rank 0][INFO] BBox mask shape: torch.Size([1, 3, 49, 480, 832])
[2025-07-15 23:10:23][Rank 0][INFO] Object masks shape: torch.Size([2, 49, 1, 480, 832])
[2025-07-15 23:10:23][Rank 0][INFO] Object bbox masks shape: torch.Size([2, 49, 1, 480, 832])
[2025-07-15 23:10:23][Rank 0][INFO] First frame shape: torch.Size([1, 480, 832, 3])
[2025-07-15 23:10:23][Rank 0][INFO] --------------------

'''