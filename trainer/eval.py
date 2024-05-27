import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils import thumos_postprocessing
from utils import *
import json
from trainer.eval_builder import EVAL
from utils import thumos_postprocessing, perframe_average_precision
from .criterion import calculate_metrics, calculate_multi_metrics
from criterions.loss import calculate_multi_loss
from time import perf_counter_ns

@EVAL.register("TAS")
class TASEvaluate(nn.Module):
    def __init__(self, cfg):
        super(TASEvaluate, self).__init__()
        self.cfg = cfg

    def forward(self, model, dataloader, logger, weights):
        device = "cuda:0"
        model.eval()   
        with torch.no_grad():

            epoch_elapsed = 0.0
            epoch_fps = 0.0

            epoch_loss_total = 0.0
            epoch_loss = { 'verb': 0.0, 'noun': 0.0 }
            epoch_metrics = { 
                'verb': { 'mof': 0.0, 'edit': 0.0, 'F1@10': 0.0, 'F1@25': 0.0, 'F1@50': 0.0, }, 
                'noun': { 'mof': 0.0, 'edit': 0.0, 'F1@10': 0.0, 'F1@25': 0.0, 'F1@50': 0.0, } 
            }

            start = time.time()
            batch_count = 0
            for rgb_input, poses, target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, poses, target = rgb_input.to(device), poses.to(device), target.to(device)
                frames_count = rgb_input.shape[1]
                batch_count += 1

                beg = perf_counter_ns()
                outputs = model(rgb_input, poses)
                end = perf_counter_ns()

                # Framerate approximation
                # NOTE: works only because test_batch_size is 1
                elapsed_ms = (end - beg) * 1e-6
                epoch_elapsed += (elapsed_ms / frames_count)
                epoch_fps += frames_count / (elapsed_ms * 1e-3)

                target_verb, target_noun = torch.split(target, split_size_or_sections=1, dim=-1)
                target_verb = torch.squeeze(target_verb, dim=-1)
                target_noun = torch.squeeze(target_noun, dim=-1)
                target = [target_verb, target_noun]

                categories = [('verb', 25), ('noun', 91)]
                
                # Criterions
                batch_metrics = calculate_multi_metrics(outputs, target, categories)
                for category_name, category_metrics in epoch_metrics.items():
                    for metric_name, metric_value in category_metrics.items():
                        epoch_metrics[category_name][metric_name] += batch_metrics[category_name][metric_name]

                # Loss
                losses, combined_loss = calculate_multi_loss(outputs, target, categories, self.cfg['alpha'], weights)
                epoch_loss['verb'] += losses['verb']['loss_total']
                epoch_loss['noun'] += losses['noun']['loss_total']
                epoch_loss_total += combined_loss

            end = time.time()
            time_taken = end - start

            logger.info(f'Processed frames in {time_taken:.1f} seconds, metrics={metrics}')

        epoch_elapsed /= batch_count
        epoch_fps /= batch_count

        logger.info(f'elapsed(ms): {epoch_elapsed}, epoch_fps: {epoch_fps}')

        # Mean over the entire epoch
        for category_name, category_value in epoch_metrics.items():
            for metric_name, _ in category_value.items():
                epoch_metrics[category_name][metric_name] /= batch_count

        epoch_loss_total /= batch_count
        epoch_loss['verb'] /= batch_count
        epoch_loss['noun'] /= batch_count

        return epoch_loss_total, epoch_loss, epoch_metrics

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
    