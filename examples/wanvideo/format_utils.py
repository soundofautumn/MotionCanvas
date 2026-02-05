import torch
from PIL import Image
import numpy as np
def tensor_video_to_pil_list(video_tensor):
    # video_tensor shape: (B, T, C, H, W), dtype=torch.uint8, range [0, 255]
    B, T, C, H, W = video_tensor.shape
    pil_batch = []
    
    # Move tensor to CPU and permute to (B, T, H, W, C)
    video_tensor = video_tensor.permute(0, 1, 3, 4, 2).numpy()

    for b in range(B):
        frame_list = []
        for t in range(T):
            frame_array = video_tensor[b, t]  # shape: (H, W, C), dtype: uint8
            pil_img = Image.fromarray(frame_array)
            frame_list.append(pil_img)
        pil_batch.append(frame_list)

    return pil_batch


def pil_list_to_tensor_video(pil_batch):
    batch_tensor = []
    for frame_list in pil_batch:
        frame_tensor_list = []
        for img in frame_list:
            img_array = np.array(img)  # (H,W,C), dtype=uint8
            frame_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # to(C,H,W)
            frame_tensor_list.append(frame_tensor)
        video_tensor = torch.stack(frame_tensor_list, dim=0)  # (T,C,H,W)
        batch_tensor.append(video_tensor)

    final_tensor = torch.stack(batch_tensor, dim=0)  # (B,T,C,H,W)
    return final_tensor

def tensor_to_video_list(video_tensor):
    """
    Convert a BTCHW torch tensor (float32, 0-255 range) to a list of video frames as PIL Images.
    
    Args:
        video_tensor: Input video tensor with shape (batch, time, channels, height, width)
                      in float32 format with values in range [0, 255].
                      
    Returns:
        A list where each element represents a video (list of PIL Images).
        
    Raises:
        ValueError: If input tensor doesn't have 5 dimensions or channels is not 3 (RGB).
    """
    # Check tensor dimensions
    if len(video_tensor.shape) != 5:
        raise ValueError(f"Expected 5D tensor (BTCHW), got {len(video_tensor.shape)}D tensor")
    
    _, _, channels, _, _ = video_tensor.shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {channels} channels")
    
    # Convert tensor to uint8 numpy array and move to CPU if needed
    video_tensor = video_tensor.to(torch.uint8).cpu()

    
    video_list = []
    for batch_video in video_tensor:
        # Process each video in the batch
        pil_frames = []
        for frame in batch_video:
            # Convert CHW tensor to HWC numpy array
            frame_np = frame.permute(1, 2, 0).numpy()
            # Create PIL Image
            pil_img = Image.fromarray(frame_np)
            pil_frames.append(pil_img)
        video_list.append(pil_frames)
    
    return video_list