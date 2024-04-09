import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import thumos_postprocessing
from utils import *
import json
from trainer.eval_builder import EVAL
from utils import thumos_postprocessing, perframe_average_precision
from .criterion import calculate_metrics

@EVAL.register("TAS")
class TASEvaluate(nn.Module):
    def __init__(self, cfg):
        super(TASEvaluate, self).__init__()

    def forward(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()   
        with torch.no_grad():

            metrics = {
                'F1@10': 0.0,
                'F1@25': 0.0,
                'F1@50': 0.0,
                'edit':  0.0,
                'mof':   0.0
            }

            start = time.time()
            batch_count = 0
            for rgb_input, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, target = rgb_input.to(device), target.to(device)
                batch_count += target.shape[0]

                # Process batch                
                out_dict = model(rgb_input, None)
                logits = out_dict['logits']
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)

                batch_metrics = calculate_metrics(predictions, target, prefix='')
                metrics = { key: val + batch_metrics[key] for key, val in metrics.items() }

            end = time.time()
            time_taken = end - start

            metrics = { key: val / batch_count for key, val in metrics.items() }
            metrics['edit'] /= 100 # NOTE: it's upper value is 100, normalize it

            logger.info(f'Processed frames in {time_taken:.1f} seconds, metrics={metrics}')

        # Sum all criterions to get final quality score
        final_score = sum(value for value in metrics.values())
        return final_score

@EVAL.register("OAD")
class Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
    
    def eval(self, model, dataloader, logger):
        device = "cuda:0"
        model.eval()   
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            for rgb_input, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, target = rgb_input.to(device), target.to(device)
                out_dict = model(rgb_input, None)

                pred_logit = out_dict['logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val) 
                gt_targets += list(target_batch)

            end = time.time()
            num_frames = len(gt_targets)
            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, self.metric)
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')

        return result['mean_AP']
    
    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)

#@EVAL.register("ANTICIPATION")
#class ANT_Evaluate(nn.Module):
#    
#    def __init__(self, cfg):
#        super(ANT_Evaluate, self).__init__()
#        data_name = cfg["data_name"].split('_')[0]
#        self.data_processing = thumos_postprocessing if data_name == 'THUMOS' else None
#        self.metric = cfg['metric']
#        self.eval_method = perframe_average_precision
#        self.all_class_names = json.load(open(cfg['video_list_path']))[data_name]['class_index']
#    
#    def eval(self, model, dataloader, logger):
#        device = "cuda:0"
#        model.eval()   
#        with torch.no_grad():
#            pred_scores, gt_targets, ant_pred_scores, ant_gt_targets = [], [], [], []
#            start = time.time()
#            anticipation_mAPs = []
#            for rgb_input, flow_input, target, ant_target in tqdm(dataloader, desc='Evaluation:', leave=False):
#                rgb_input, flow_input, target, ant_target = rgb_input.to(device), flow_input.to(device), target.to(device), ant_target.to(device)
#                out_dict = model(rgb_input, flow_input)
#                pred_logit = out_dict['logits']
#                ant_pred_logit = out_dict['anticipation_logits']
#                prob_val = pred_logit.squeeze().cpu().numpy()
#                target_batch = target.squeeze().cpu().numpy()
#                ant_prob_val = ant_pred_logit.squeeze().cpu().numpy()
#                ant_target_batch = ant_target.squeeze().cpu().numpy()
#                pred_scores += list(prob_val)  
#                gt_targets += list(target_batch)
#                ant_pred_scores += list(ant_prob_val)
#                ant_gt_targets += list(ant_target_batch)      
#            end = time.time()
#            num_frames = len(gt_targets)
#            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, self.metric)
#            ant_pred_scores = np.array(ant_pred_scores)
#            ant_gt_targets = np.array(ant_gt_targets)
#            logger.info(f'OAD mAP: {result["mean_AP"]*100:.2f}')
#            for step in range(ant_gt_targets.shape[1]):
#                result[f'anticipation_{step+1}'] = self.eval_method(ant_pred_scores[:,step,:], ant_gt_targets[:,step,:], self.all_class_names, self.data_processing, self.metric)
#                anticipation_mAPs.append(result[f'anticipation_{step+1}']['mean_AP'])
#                logger.info(f"Anticipation at step {step+1}: {result[f'anticipation_{step+1}']['mean_AP']*100:.2f}")
#            logger.info(f'Mean Anticipation mAP: {np.mean(anticipation_mAPs)*100:.2f}')
#            
#            time_taken = end - start
#            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
#            
#        return np.mean(anticipation_mAPs)
    
    def forward(self, model, dataloader, logger):
        return self.eval(model, dataloader, logger)
    