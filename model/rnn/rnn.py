import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_builder import META_ARCHITECTURES
from time import perf_counter_ns
import einops

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048,
    'rgb_TSM': 2048
}

@META_ARCHITECTURES.register("MiniROAD")
class MROAD(nn.Module):
    
    def __init__(self, cfg):
        super(MROAD, self).__init__()
        self.modality = cfg['modality']

        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.out_dim = cfg['num_classes']
        self.window_size = cfg['window_size']

        self.relu = nn.ReLU()
        self.embedding_dim = cfg['embedding_dim']
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # for poses
        if self.modality == 'poses':
            self.input_dim = 42 * 3
        # for frames
        else:
            self.input_dim = FEATURE_SIZES[cfg['rgb_type']]

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )

        self.f_classification_verb = nn.Sequential(
            nn.Linear(self.hidden_dim, 25) # 24 + 1 background
        )
        self.f_classification_noun = nn.Sequential(
            nn.Linear(self.hidden_dim, 91) # 90 + 1 background
        )

        # self.h0 = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

    def benchmark_framerate(self):
        bob = torch.zeros((1, 2, 2048)).to('cuda')

        fps = 0.0
        elapsed = 0.0
        for _ in range(1000):
            
            beg = perf_counter_ns()
            _ = self.forward(bob, None)
            end = perf_counter_ns()

            elapsed_ms = (end - beg) * 1e-6
            elapsed += elapsed_ms
            fps += 1.0 / (elapsed_ms * 1e-3)

        elapsed /= 1000.0
        fps /= 1000.0

        return elapsed, fps

    def forward(self, rgb_input, poses):
        
        #if self.use_rgb and self.use_flow:
        #    x = torch.cat((rgb_input, flow_input), 2)
        #elif self.use_rgb:
        #    x = rgb_input
        #elif self.use_flow:
        #    x = flow_input

        if self.modality == 'poses':
            x = einops.rearrange(poses, 'b t j f -> b t (j f)')
        else:
            x = rgb_input

        x = self.layer1(x)
        
        B, _, _ = x.shape
        h0 = self.h0.expand(-1, B, -1).to(x.device)
        ht, _ = self.gru(x, h0) 
        ht = self.relu(ht)
        # ht = self.relu(ht + x)
        
        #logits = self.f_classification(ht)
        logits_verbs = self.f_classification_verb(ht)
        logits_nouns = self.f_classification_noun(ht)
        return [logits_verbs, logits_nouns]

        #if self.training:
        #    out_dict['logits_verbs'] = logits_verbs
        #    out_dict['logits_verbs'] = logits_verbs
        #else:
        #    pred_scores = F.softmax(logits, dim=-1)
        #    out_dict['logits'] = pred_scores
        #return out_dict

@META_ARCHITECTURES.register("MiniROADA")
class MROADA(nn.Module):
    
    def __init__(self, cfg):
        super(MROADA, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.embedding_dim = cfg['embedding_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.anticipation_length = cfg["anticipation_length"]
        self.out_dim = cfg['num_classes']

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout'])
        )
        self.actionness = cfg['actionness']
        if self.actionness:
            self.f_actionness = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
            )
        self.relu = nn.ReLU()
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.anticipation_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.anticipation_length*self.hidden_dim),
        )

    def forward(self, rgb_input, flow_input):
        if self.use_rgb and self.use_flow:
            x = torch.cat((rgb_input, flow_input), 2)
        elif self.use_rgb:
            x = rgb_input
        elif self.use_flow:
            x = flow_input
        B, S, _ = x.shape
        x = self.layer1(x)
        h0 = torch.zeros(1, B, self.hidden_dim).to(x.device)
        ht, _ = self.gru(x, h0)
        logits = self.f_classification(self.relu(ht))
        anticipation_ht = self.anticipation_layer(self.relu(ht)).view(B, S, self.anticipation_length, self.hidden_dim)
        anticipation_logits = self.f_classification(self.relu(anticipation_ht))
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
            out_dict['anticipation_logits'] = anticipation_logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            pred_anticipation_scores = F.softmax(anticipation_logits, dim=-1)
            out_dict['logits'] = pred_scores
            out_dict['anticipation_logits'] = pred_anticipation_scores

        return out_dict