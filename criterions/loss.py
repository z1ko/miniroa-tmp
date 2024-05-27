import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS
import einops as ein

def calculate_multi_loss(logits, targets, categories, alpha, weights):
    """ Calculate ce+mse multiloss between different target categories
            logits:     an array of tensors, each of the shape [batch_size, seq_len, logits]
            targets:    an array of tensors, each of the shape [batch_size, seq_len]
            categories: [('verb', 25), ('noun', 90)]
    """

    result = { }
    for category_name, _ in categories:
        result[category_name] = {}

    combined = 0.0
    for i, (logit, target, category) in enumerate(zip(logits, targets, categories)):
        #assert(target.shape[0] == 1)
        category_name, num_classes = category
        loss = CEplusMSE(num_classes, alpha=alpha, weight=weights[i])
        category_result = loss(logit, target)
        result[category_name] = category_result
        
        # Accumulated loss between all categories
        combined += category_result['loss_total']

    return result, combined

@CRITERIONS.register('CEplusMSE')
class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """

    def __init__(self, num_classes, weight, alpha=0.17):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.mse = nn.MSELoss(reduction='none')
        self.classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits:  [batch_size, seq_len, logits]
        :param targets: [batch_size, seq_len]
        """

        logits = ein.rearrange(logits, 'batch_size seq_len logits -> batch_size logits seq_len')
        loss = { }

        # Frame level classification
        loss['loss_ce'] = self.ce(
            ein.rearrange(logits, "batch_size logits seq_len -> (batch_size seq_len) logits"),
            ein.rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        )

        # Neighbour frames should have similar values
        loss['loss_mse'] = torch.mean(torch.clamp(self.mse(
            F.log_softmax(logits[:, :, 1:], dim=1),
            F.log_softmax(logits.detach()[:, :, :-1], dim=1)
        ), min=0.0, max=160.0))

        loss['loss_total'] = loss['loss_ce'] + self.alpha * loss['loss_mse']
        return loss

@CRITERIONS.register('NONUNIFORM')
class OadLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target): 
        return self.loss(out_dict, target)
    

@CRITERIONS.register('ANTICIPATION')
class OadAntLoss(nn.Module):
    
    def __init__(self, cfg, reduction='sum'):
        super(OadAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['num_classes']

    def anticipation_loss(self, out_dict, target, ant_target):
        anticipation_logits = out_dict['anticipation_logits']
        pred_anticipation_logits = anticipation_logits[:,-1,:,:].contiguous().view(-1, self.num_classes)
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)
        return ant_loss 

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target): 
        return self.loss(out_dict, target, ant_target)
