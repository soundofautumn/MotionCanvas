from collections import defaultdict
import json
import numpy as np
from pycocotools import mask as mask_utils
import ossutils as ops


class MaskLoader:
    @staticmethod
    def _get_from_dict(data):
        video_segments = data['annotations']
        id_to_names = data['id_to_names']
        obj_mask_map = defaultdict(dict)  
        
        frame_num = len(video_segments)
        
        for obj_id in id_to_names.keys():
            obj_mask_map[obj_id] = {
                'masks': [],
                'areas': [],
                'boxes': []
            }
        
        for frame_idx, segments in video_segments.items():
            for obj_id, info in segments.items():
                # handle case where obj_id is in id_to_names
                if obj_id in id_to_names:
                    seg = np.array(mask_utils.decode(info['mask'])).astype(np.uint8)
                    mask_area = info['mask_area']
                    mask_box = info['mask_box']  # [x1, y1, x2, y2]
                    
                    obj_mask_map[obj_id]['masks'].append(seg)
                    obj_mask_map[obj_id]['areas'].append(mask_area)
                    obj_mask_map[obj_id]['boxes'].append(mask_box)
        
        new_map = {}
        for obj_id, obj_info in obj_mask_map.items():
            if len(obj_info['masks']) == frame_num:
                new_map[obj_id] = {
                    'masks': np.stack(obj_info['masks'], axis=0),  # [T, H, W]
                    'areas': np.array(obj_info['areas']),          # [T]
                    'boxes': np.array(obj_info['boxes'])           # [T, 4]
                }
        
        return id_to_names, new_map

    def __call__(self, mask):
        return self._get_from_dict(mask)


class MaskJsonLoader(MaskLoader):
    def __call__(self, mask):
        assert isinstance(mask, str)
        if mask.startswith('oss://'):
            data = json.loads(ops.read(mask, mode='r'))
        else:
            data = json.loads(mask)
        return self._get_from_dict(data)