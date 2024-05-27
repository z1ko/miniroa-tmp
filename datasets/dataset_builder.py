__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch
import torch.utils.data as data
from utils import Registry

DATA_LAYERS = Registry()

def build_dataset(cfg):
    data_layer = DATA_LAYERS[f'{cfg["data_name"]}']
    return data_layer

def collate(data):
        embeddings, poses, labels = [], [], []
        for e, p, l in data:
            embeddings.append(e)
            poses.append(p)
            labels.append(l)
        
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        poses = torch.nn.utils.rnn.pad_sequence(poses, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return embeddings, poses, labels

def build_data_loader(cfg, mode):
    data_layer = build_dataset(cfg)

    collate_fn = collate if cfg["data_name"] == "Assembly101" and mode != "train" else None
    data_loader = data.DataLoader(
        dataset=data_layer(cfg, mode),
        batch_size=cfg["batch_size"] if mode == 'train' else cfg["test_batch_size"],
        shuffle=True if mode == 'train' else False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return data_loader