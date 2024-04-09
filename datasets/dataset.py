import torch
import torch.utils.data as data
import numpy as np
import einops as ein
import json
import tqdm
import os.path as osp
import pandas as pd
import gc
import os
import lmdb
from datasets.dataset_builder import DATA_LAYERS 

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048
}

@DATA_LAYERS.register("THUMOS")
@DATA_LAYERS.register("TVSERIES")
class THUMOSDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()
        
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        self.target_all = {}
        self.rgb_inputs = {}
        #self.flow_inputs = {}
        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['rgb_type']]))
        #dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['flow_type']]))
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            #flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
            # concatting dummy target at the front 
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                #self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
            else:
                self.target_all[vid] = target
                self.rgb_inputs[vid] = rgb
                #self.flow_inputs[vid] = flow
    
    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []
        for vid in self.vids:
            target = self.target_all[vid]
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end]
                    ])
            else:
                start = 0
                end = target.shape[0]
                self.inputs.append([
                    vid, start, end, target[start:end]
                ])

    def __getitem__(self, index):
        vid, start, end, target = self.inputs[index]
        rgb_input = self.rgb_inputs[vid][start:end]
        #flow_input = self.flow_inputs[vid][start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        #flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        return rgb_input, target

    def __len__(self):
        return len(self.inputs)

@DATA_LAYERS.register("Assembly101")
class AssemblyDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        
        self.mode = 'train' if mode == 'train' else 'val'

        self.views = ['C10095_rgb']
        self.window_size = cfg['window_size']

        # Load all view databases
        self.path_to_data = 'data/Assembly101'
        self.envs = {view: lmdb.open(f'{self.path_to_data}/TSM/{view}', 
                                     readonly=True, 
                                     lock=False) for view in self.views}
        
        # Load actions and data
        actions_path = os.path.join(self.path_to_data, 'coarse-annotations', 'actions.csv')
        self.actions = pd.read_csv(actions_path)
        self.data = self.make_database()

    def make_database(self):
        annotations_path = os.path.join(self.path_to_data, 'coarse-annotations')
        data_path = os.path.join(self.path_to_data, f'{self.mode}.csv')
        data_df = pd.read_csv(data_path)

        video_data = []
        max_len, min_len = -1, 1e6
        for _, entry in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
            sample = entry.to_dict()
            
            # Skip views no required
            if sample['view'] not in self.views:
                continue

            segm_filename = f"{sample['action_type']}_{sample['video_id']}.txt"
            segm_path = os.path.join(annotations_path, "coarse_labels", segm_filename)
            segm, start_frame, end_frame = self.load_segmentation(segm_path, self.actions)

            max_len = max(max_len, end_frame - start_frame)
            min_len = min(min_len, end_frame - start_frame)

            # Create subsections of window_size for training
            for beg in range(0, len(segm) - self.window_size, self.window_size):
                end = beg + self.window_size

                sample['segm'] = torch.tensor(segm[beg:end]).long()
                sample['start_frame'] = start_frame + beg
                sample['end_frame'] = start_frame + end
                video_data.append(sample)

            a = 0

        print(f'elements={len(video_data)}, max_frames={max_len}, min_frames={min_len}')
        return video_data

    def load_segmentation(self, segm_path, actions):
        labels = []
        start_indices = []
        end_indices = []

        with open(segm_path, 'r') as f:
            lines = list(map(lambda s: s.split("\n"), f.readlines()))
            for line in lines:
                start, end, lbl = line[0].split("\t")[:-1]
                start_indices.append(int(start))
                end_indices.append(int(end))
                action_id = actions.loc[actions['action_cls'] == lbl, 'action_id']
                segm_len = int(end) - int(start)
                labels.append(np.full(segm_len, fill_value=action_id.item()))

        segmentation = np.concatenate(labels)
        num_frames = segmentation.shape[0]

        # start and end frame idx @30fps
        start_frame = min(start_indices)
        end_frame = max(end_indices)
        assert num_frames == (end_frame-start_frame), \
            "Length of Segmentation doesn't match with clip length."

        return segmentation, start_frame, end_frame

    def load_features(self, data_dict):
        elements = []
        view = data_dict['view']
        with self.envs[view].begin(write=False) as e:
            for i in range(data_dict['start_frame'], data_dict['end_frame']):
                key = os.path.join(data_dict['video_id'], f'{view}/{view}_{i:010d}.jpg')
                frame_data = e.get(key.strip().encode('utf-8'))
                if frame_data is None:
                    print(f"[!] No data found for key={key}.")
                    exit(2)

                frame_data = np.frombuffer(frame_data, 'float32')
                elements.append(frame_data)

        features = np.array(elements) # [T, D]
        return features

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        targets = data_dict['segm']
        features = torch.tensor(self.load_features(data_dict))
        return features, targets

    def __len__(self):
        return len(self.data)

@DATA_LAYERS.register("THUMOS_ANTICIPATION")
@DATA_LAYERS.register("TVSERIES_ANTICIPATION")
class THUMOSDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        self.anticipation_length = cfg['anticipation_length']
        data_name = cfg["data_name"].split('_')[0]
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()
        
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        self.target_all = {}
        self.rgb_inputs = {}
        self.flow_inputs = {}
        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['rgb_type']]))
        dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['flow_type']]))
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
            else:
                self.target_all[vid] = target
                self.rgb_inputs[vid] = rgb
                self.flow_inputs[vid] = flow
        
    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []

        for vid in self.vids:
            target = self.target_all[vid]
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]-self.anticipation_length, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end], target[end:end+self.anticipation_length]
                    ])
            else:
                start = 0
                end = target.shape[0] - self.anticipation_length
                ant_target = []
                for s in range(0, target.shape[0]-self.anticipation_length):
                    ant_target.append(target[s:s+self.anticipation_length])

                self.inputs.append([
                    vid, start, end, target[start:end], np.array(ant_target)
                ])
    
    def __getitem__(self, index):
        vid, start, end, target, ant_target = self.inputs[index]
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        ant_target = torch.tensor(ant_target.astype(np.float32))
        return rgb_input, flow_input, target, ant_target

    def __len__(self):
        return len(self.inputs)
   
@DATA_LAYERS.register("FINEACTION")
class FINEACTIONDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set'] # list of video names
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()

    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        
    def _init_features(self, seed=0):
        # self.inputs = []
        del self.inputs
        gc.collect()
        self.inputs = []
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end
                    ])
            else:
                start = 0
                end = target.shape[0]
                self.inputs.append([
                    vid, start, end
                ])

    def __getitem__(self, index):
        vid, start, end = self.inputs[index]
        rgb_input = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'), mmap_mode='r')[start:end]
        flow_input = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'), mmap_mode='r')[start:end]
        target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'), mmap_mode='r')[start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        return rgb_input, flow_input, target

    def __len__(self):
        return len(self.inputs)    