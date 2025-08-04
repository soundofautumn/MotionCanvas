import os
import sys
sys.path.append(os.path.abspath('.'))
import random
import numpy as np
import torch
import oss2
import decord
from decord import VideoReader
from io import BytesIO


SKIP_ZERO = True

def get_pos_emb(
    pos_k: torch.Tensor,
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate batch position embeddings.
    
    Args:
        pos_k (torch.Tensor): A 1D tensor containing positions for which to generate embeddings.
        pos_emb_dim (int): The dimension of position embeddings.
        theta_func (callable): Function to compute thetas based on position and embedding dimensions.
        device (torch.device): Device to store the position embeddings.
        dtype (torch.dtype): Desired data type for computations.
    
    Returns:
        torch.Tensor: The position embeddings with shape (batch_size, pos_emb_dim).
    """
    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device=device, dtype=dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
    # Expand denominator to match the shape needed for broadcasting
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)
    
    thetas = theta_func(denominator_expanded, pos_emb_dim)
    
    # Ensure pos_k is in the correct shape for broadcasting
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

    # Concatenate sine and cosine embeddings along the last dimension
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb


def create_pos_feature_map(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    downsample_ratios: list[int],
    height: int,
    width: int,
    pos_emb_dim: int,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    assert t_down_strategy in ["sample", "average"], "Invalid time downsampling strategy"
    
    B, T, N, _ = pred_tracks.shape
    t_down, h_down, w_down = downsample_ratios
    T_prime = (T - 1) // t_down + 1
    H_prime = height // h_down
    W_prime = width // w_down
    
    feature_map = torch.zeros(B, T_prime, H_prime, W_prime, pos_emb_dim, 
                             device=device, dtype=dtype)
    
    if track_num == -1:
        track_num = N
    
    # Track sampling
    tracks_idx = torch.stack([
        torch.randperm(N, device=device)[:track_num] 
        for _ in range(B)
    ])
    
    # Get position embeddings
    global_embs = get_pos_emb(torch.arange(N, device=device), pos_emb_dim, device=device, dtype=dtype)
    tracks_embs = global_embs[tracks_idx]  # [B, track_num, pos_emb_dim]
    
    # Time downsampling (same as original)
    t_indices = torch.arange(0, T, t_down, device=device)[:T_prime]
    if t_down_strategy == "sample":
        # Direct sampling: [B, T_prime, track_num, 2]
        sampled_tracks = pred_tracks[:, t_indices]
        sampled_visibility = pred_visibility[:, t_indices]
    else:  # "average"
        # Pre-calculate valid time windows
        time_windows = []
        for t_idx in t_indices:
            end = min(t_idx + t_down, T)
            if t_idx == 0:  # First frame always uses sample
                window = slice(t_idx, t_idx + 1)
            else:
                window = slice(t_idx, end)
            time_windows.append(window)
        
        # Process each time window
        sampled_tracks = []
        sampled_visibility = []
        
        for window in time_windows:
            # For visibility: OR operation over the window
            vis_window = torch.any(
                pred_visibility[:, window], 
                dim=1,
                keepdim=True
            )  # [B, 1, N]
            
            # For positions: average over visible frames
            pos_window = pred_tracks[:, window]  # [B, W, N, 2]
            
            # Compute mean while ignoring invalid positions
            valid_mask = pred_visibility[:, window].unsqueeze(-1)  # [B, W, N, 1]
            pos_sum = (pos_window * valid_mask).sum(dim=1)  # [B, N, 2]
            valid_count = valid_mask.sum(dim=1).clamp(min=1)  # [B, N, 1]
            avg_pos = pos_sum / valid_count
            
            sampled_tracks.append(avg_pos.unsqueeze(1))
            sampled_visibility.append(vis_window.unsqueeze(1))
        
        sampled_tracks = torch.cat(sampled_tracks, dim=1)      # [B, T_prime, N, 2]
        sampled_visibility = torch.cat(sampled_visibility, dim=1)  # [B, T_prime, N]
    
    # Select sampled tracks [B, T_prime, track_num, 2]
    batch_idx = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
    time_idx = torch.arange(T_prime, device=device)[None, :, None]  # [1, T_prime, 1]
    sampled_tracks = sampled_tracks[batch_idx, time_idx, tracks_idx.unsqueeze(1)]
    sampled_visibility = sampled_visibility[batch_idx, time_idx, tracks_idx.unsqueeze(1)]

    # ===== 3. Coordinate Transformation =====
    # Convert coordinates to feature map space [B, T_prime, track_num]
    x_coords = (sampled_tracks[..., 0] / w_down).long()  # x in [0, W_prime-1]
    y_coords = (sampled_tracks[..., 1] / h_down).long()  # y in [0, H_prime-1]
    
    # Create valid mask [B, T_prime, track_num]
    valid_mask = sampled_visibility & \
                 (x_coords >= 0) & (x_coords < W_prime) & \
                 (y_coords >= 0) & (y_coords < H_prime)
    
    # ===== 4. Scatter Updates =====
    # Get indices of valid points
    b_idx, t_idx, tr_idx = torch.where(valid_mask)
    
    # Get corresponding coordinates and embeddings
    x_valid = x_coords[b_idx, t_idx, tr_idx]
    y_valid = y_coords[b_idx, t_idx, tr_idx]
    embeddings_valid = tracks_embs[b_idx, tr_idx]  # [n_valid, pos_emb_dim]
    
    # Scatter-add to feature map
    feature_map.index_put_(
        indices=(b_idx, t_idx, y_valid, x_valid),
        values=embeddings_valid,
        accumulate=True
    )
    
    return feature_map, None


def replace_feature_batch_optimized(
    vae_feature: torch.Tensor,  # [B, C', T', H', W']
    track_pos: torch.Tensor,    # [B, N, T', 2]
) -> torch.Tensor:
    b, _, t, h, w = vae_feature.shape
    assert b == track_pos.shape[0], "Batch size mismatch."
    n = track_pos.shape[1]
    
    track_pos = track_pos[:, torch.randperm(n), :, :]
    
    current_pos = track_pos[:, :, 1:, :]  # [B, N, T-1, 2]
    mask = (current_pos[..., 0] >= 0) & (current_pos[..., 1] >= 0)  # [B, N, T-1]
    
    valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 3]
    num_valid = valid_indices.shape[0]
    
    if num_valid == 0:
        return vae_feature
    
    batch_idx = valid_indices[:, 0]
    track_idx = valid_indices[:, 1]
    t_rel = valid_indices[:, 2]
    t_target = t_rel + 1  
    
    # 
    h_target = current_pos[batch_idx, track_idx, t_rel, 0].long()  
    w_target = current_pos[batch_idx, track_idx, t_rel, 1].long()
    
    h_source = track_pos[batch_idx, track_idx, 0, 0].long()
    w_source = track_pos[batch_idx, track_idx, 0, 1].long()
    
    src_features = vae_feature[batch_idx, :, 0, h_source, w_source]
    vae_feature[batch_idx, :, t_target, h_target, w_target] = src_features
    
    return vae_feature


def get_video_track_video(
    model,
    video_tensor: torch.Tensor,
    object_masks: list[torch.Tensor],
    downsample_ratios: list[int],
    pos_emb_dim: int,
    grid_size: int = 12,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    random_sample_p: float = 0.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
    logger: object = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    b, t, c, height, width = video_tensor.shape
    n_queries = grid_size * grid_size
    obj_assignments = []  # Store object assignments per sample
    
    # Function to assign object indices to points
    def assign_obj_indices(obj_masks):
        num_objs = obj_masks.shape[0]
        assignment = torch.full((height, width), -1, dtype=torch.long, device=device)
        
        # Create assignment map: which object each pixel belongs to
        # unassigned = torch.ones((height, width), dtype=torch.bool, device=device)
        for obj_idx in range(num_objs):
            # mask = obj_masks[obj_idx, 0, 0] & unassigned
            mask = obj_masks[obj_idx, 0, 0]
            assignment[mask] = obj_idx
            # unassigned = unassigned & ~mask
        
        return assignment

    # Branch 1: Random grid sampling
    if random.random() < random_sample_p:
        # queries_list = []
        for i in range(b):
            obj_masks = object_masks[i]  # [num_objs, T, 1, H, W]
            
            # Get object assignments for this sample
            assignment = assign_obj_indices(obj_masks)
            obj_assignments.append(assignment)
        
        # queries_batch = torch.stack(queries_list, dim=0).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size, backward_tracking=False) #B T N 2,  B T N 1
    
    # Branch 2: Object-aware sampling
    else:
        queries_list = []
        for i in range(b):
            obj_masks = object_masks[i]  # [num_objs, T, 1, H, W]
            
            # Get object assignments for this sample
            assignment = assign_obj_indices(obj_masks)
            obj_assignments.append(assignment)
            
            # Create merged mask
            merged_mask = obj_masks.sum(dim=0) > 0
            first_frame_mask = merged_mask[0, 0]
            
            # Get foreground coordinates
            try:
                fg_coords = (first_frame_mask > 0).nonzero(as_tuple=True)
            except Exception as e:
                # 
                logger.info("first_frame_mask shape:", first_frame_mask.shape)
                logger.info("first_frame_mask dtype:", first_frame_mask.dtype)
                logger.info("first_frame_mask device:", first_frame_mask.device)

                # 
                logger.info("Unique values:", torch.unique(first_frame_mask))
                logger.info("NaN count:", torch.isnan(first_frame_mask).sum().item())
                logger.info("Inf count:", torch.isinf(first_frame_mask).sum().item())

                # 
                if (first_frame_mask > 0).any():
                    logger.info("Valid mask detected")
                else:
                    logger.info("WARNING: All values <=0 in mask!") 
                
                raise e
            
            if len(fg_coords[0]) == 0:
                # Fallback to whole image sampling
                y_coords = torch.randint(0, height, (n_queries,)).to(device)
                x_coords = torch.randint(0, width, (n_queries,)).to(device)
            else:
                num_fg = len(fg_coords[0])
                indices = torch.randint(0, num_fg, (n_queries,))
                y_coords = fg_coords[0][indices]
                x_coords = fg_coords[1][indices]
            
            t_vals = torch.zeros(n_queries, device=device)
            sample_queries = torch.stack([t_vals, x_coords.float(), y_coords.float()], dim=1)
            queries_list.append(sample_queries)
        
        queries_batch = torch.stack(queries_list, dim=0).to(device)
        with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            pred_tracks, pred_visibility = model(video_tensor, queries=queries_batch, backward_tracking=False)
    
    # Stack assignments for batch processing
    obj_assignments_batch = torch.stack(obj_assignments, dim=0)  # [B, H, W]
    pred_tracks, pred_visibility = random_drop_tracks_fully(pred_tracks, pred_visibility)
    track_video, track_pos = create_pos_feature_map(
        pred_tracks, 
        pred_visibility, 
        downsample_ratios, 
        height, 
        width, 
        pos_emb_dim,
        track_num, 
        t_down_strategy, 
        device, 
        dtype
    )

    return track_video.permute(0, 4, 1, 2, 3), track_pos, pred_tracks



def resize_tracks(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    t_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize tracks to a specified number of frames.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - t_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - resized_tracks: torch.Tensor, the resized tracks, [target_frame_num, N, 2]
    - resized_visibility: torch.Tensor, the resized visibility, [target_frame_num, N]
    """

    assert t_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."
    assert t_strategy in ["sample"], "only support sample strategy."

    def get_xy_from_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the x and y coordinates from the height and width tensor.

        Args:
        - hw_tensor: torch.Tensor, the tensor of height and width, [N, height, width]

        Returns:
        - xy_tensor: torch.Tensor, the tensor of x and y coordinates, [N, 2]
        """

        h, w = hw_tensor.shape[-2:]
        _, y, x = torch.nonzero(hw_tensor, as_tuple=True)
        xy_tensor = torch.stack((x, y), dim=-1)
        assert xy_tensor.shape[0] == hw_tensor.shape[0], "The number of points should be the same."
        return xy_tensor

    def get_average_xy_from_batch_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        for b in range(hw_tensor.shape[0]):
            xy_tensor = get_xy_from_hw(hw_tensor[b])
            if b == 0:
                xy_tensors = xy_tensor
            else:
                xy_tensors += xy_tensor
        xy_tensors /= hw_tensor.shape[0]
        return xy_tensors
    
    # Get the number of frames in the input tracks
    num_frames, num_tracks, _, _ = img_tracks.shape

    new_tracks = torch.zeros(target_frame_num, num_tracks, 2, device=device, dtype=dtype)
    new_visibility = torch.ones(target_frame_num, num_tracks, device=device, dtype=torch.bool)

    new_tracks[0] = get_xy_from_hw(img_tracks[0])
    # -1 for removing the first frame
    num_frames -= 1
    target_frame_num -= 1

    new_frame_idx = 1
    if target_frame_num <= num_frames:
        t_down = num_frames / target_frame_num
        frame_idxs = [int((i - 1) * t_down + 1) for i in range(1, target_frame_num + 1)]
        for i, frame_idx in enumerate(frame_idxs):
            if t_strategy == "sample":
                new_tracks[new_frame_idx] = get_xy_from_hw(img_tracks[frame_idx])
            else:
                next_frame_idx = frame_idxs[i + 1] if i + 1 < len(frame_idxs) else num_frames + 1 # +1 as compensation for the -1
                new_tracks[new_frame_idx] = get_average_xy_from_batch_hw(img_tracks[frame_idx:next_frame_idx])

            new_frame_idx += 1
    else:
        t_repeat = target_frame_num / num_frames
        target_frame_idxs = [int((i - 1) * t_repeat + 1) for i in range(1, num_frames + 1)]
        for i, target_frame_idx in enumerate(target_frame_idxs):
            next_target_frame_idx = target_frame_idxs[i + 1] if i + 1 < len(target_frame_idxs) else target_frame_num + 1
            if t_strategy == "sample":
                new_tracks[target_frame_idx:next_target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
            else:
                if target_frame_idx == next_target_frame_idx:
                    new_tracks[target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
                else:
                    next_new_frame_idx = new_frame_idx + 1 if new_frame_idx + 1 < num_frames else new_frame_idx
                    for j in range(target_frame_idx, next_target_frame_idx):
                        new_tracks[j] = (1 - (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx)) * get_xy_from_hw(img_tracks[new_frame_idx]) + (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx) * get_xy_from_hw(img_tracks[next_new_frame_idx])

            new_frame_idx += 1

    # print(new_tracks[1])
    return new_tracks, new_visibility

def generate_custom_feature_map(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    downsample_ratios: list[int],
    pos_emb_dim: int,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a custom feature map from the tracks.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - downsample_ratios: List[int], the ratios for downsampling time, height, and width
    - pos_emb_dim: int, the dimension of the position embeddings
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - feature_map: torch.Tensor, the feature map, [T', H', W', pos_emb_dim]
    """

    height, width = img_tracks.shape[-2:]
    resized_tracks, resized_visibility = resize_tracks(img_tracks, target_frame_num, t_down_strategy, device, dtype)
    feature_map, track_pos = create_pos_feature_map(
        resized_tracks, 
        resized_visibility, 
        downsample_ratios, 
        height, 
        width, 
        pos_emb_dim, 
        track_num=-1, 
        t_down_strategy=t_down_strategy, 
        device=device, 
        dtype=dtype
    )

    return feature_map, track_pos

def random_drop_tracks_fully(pred_tracks, pred_visibility, min_keep=1):
    """
    Randomly drop between 0 to N tracks (completely remove them, reducing N).

    Args:
        pred_tracks (torch.Tensor): Shape [B, T, N, 2], tracked points.
        pred_visibility (torch.Tensor): Shape [B, T, N, 1], visibility flags.
        min_keep (int): Minimum number of tracks to keep (default=1, to avoid empty output).

    Returns:
        (torch.Tensor, torch.Tensor): Filtered tracks and visibility with reduced N.
    """
    B, T, N, _ = pred_tracks.shape

    # 1. Randomly choose how many tracks to keep (M can range from 0 to N)
    M = random.randint(min_keep, N)  # At least keep `min_keep` tracks

    # 2. Randomly select M tracks to keep (without replacement)
    keep_indices = torch.randperm(N, device=pred_tracks.device)[:M]  # [M]

    # 3. Filter the tracks and visibility
    filtered_tracks = pred_tracks[:, :, keep_indices, :]       # [B, T, M, 2]
    filtered_visibility = pred_visibility[:, :, keep_indices]  # [B, T, M, 1]

    return filtered_tracks, filtered_visibility

def set_seed(seed):
    # Set seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for random
    random.seed(seed)
    # Set seed for other libraries if needed
    # Enable deterministic mode and disable cudnn benchmarking
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_value = 0
    set_seed(seed_value)

    device = torch.device("cuda")
    dtype = torch.float32
    downsample_ratios = [4, 8, 8]
    cotracker = torch.hub.load("/mnt/29a30e4acb8/jinboxing/cache/torch_cache/hub/facebookresearch_co-tracker_main", "cotracker3_offline", source="local").to(device, dtype=dtype)

    auth = oss2.Auth(os.getenv('OSS_AK'), os.getenv('OSS_SK'))
    bucket = oss2.Bucket(auth, 'http://oss-cn-wulanchabu-internal.aliyuncs.com', 'video-generation-wulanchabu')
    video_path = "xxx.mp4"

    def get_object(bucket, video_path, reader=lambda u: u, retry=5):
        error = None
        for _ in range(retry):
            try:
                return reader(bucket.get_object(video_path).read())
            except Exception as e:
                error = e
                continue
        else:
            print(f'get_object from {video_path} failed with error: {error}')

    decord.bridge.set_bridge('torch')
    reader = VideoReader(get_object(bucket, video_path, BytesIO))
    video_tensor = reader.get_batch(range(33)).permute(0, 3, 1, 2).to(device, dtype=dtype) # [T, C, H, W]

    pos_emb_dim = 64
    track_video, track_pos = get_video_track_video(cotracker, video_tensor.unsqueeze(0), None, downsample_ratios, pos_emb_dim, random_sample_p=1, device=device, dtype=dtype)
    print(track_video.shape)
    print(track_video)

    t_down, h_down, w_down = downsample_ratios
    _, t1, h1, w1 = track_video.shape
    vae_feature = torch.randn(pos_emb_dim, t1, h1, w1, device=device, dtype=dtype)
    vae_feature = replace_feature_batch_optimized(vae_feature.unsqueeze(0).repeat(2, 1, 1, 1, 1), track_pos.unsqueeze(0).repeat(2, 1, 1, 1))
    print(vae_feature.shape)
    
    test_replace_count = 0
    for b in range(vae_feature.shape[0]):
        for i in range(track_pos.shape[0]):
            for j in range(1, track_pos.shape[1]):
                y_0, x_0 = track_pos[i, 0]
                y, x = track_pos[i, j]
                if y < 0 or x < 0:
                    continue
                if vae_feature[b, :, j, y, x].ne(vae_feature[b, :, 0, y_0, x_0]).any():
                    print(b, i, j)
                    test_replace_count +=1
    print(test_replace_count)
