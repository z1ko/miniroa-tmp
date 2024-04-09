from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ein
import numpy as np

def calculate_metrics(predictions, targets, prefix='val/'):

    mof = MeanOverFramesAccuracy()
    f1 = F1Score()
    edit = EditDistance(True)

    result = { 'mof': mof(predictions, targets), 'edit': edit(predictions, targets) }
    result.update(f1(predictions, targets))
    result = { f'{prefix}{key}': val for key,val in result.items() }
    return result

def _get_labels_start_end_time(frame_wise_labels, ignored_classes=[-100]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in ignored_classes:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in ignored_classes:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in ignored_classes:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in ignored_classes:
        ends.append(i + 1)
    return labels, starts, ends


#class CEplusMSE2(nn.Module):
#    """
#    Loss from MS-TCN paper. CrossEntropy + MSE
#    https://arxiv.org/abs/1903.01945
#    """
#    def __init__(self):
#        super(CEplusMSE2, self).__init__()
#        ignore_idx = -100
#        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)
#        self.mse = nn.MSELoss(reduction='none')
#        self.mse_fraction = 0.20
#        self.mse_clip_val = 16.0
#        self.num_classes = 3
#
#    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
#        """
#        :param logits: [n_stages, batch_size, n_classes, seq_len]
#        :param targets: [batch_size, seq_len]
#        :return:
#        """
#        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
#        for p in logits:
#            loss_dict['loss_ce'] += self.ce(ein.rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
#                                            ein.rearrange(targets, "b seq_len -> (b seq_len)"))
#
#            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
#                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
#                                                            min=0,
#                                                            max=self.mse_clip_val))
#
#        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']
#        return loss_dict

class MeanOverFramesAccuracy:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())

        # Skip all padding
        mask = np.logical_not(np.isin(targets, [-100]))

        total = mask.sum()
        correct = (predictions == targets)[mask].sum()
        result = correct / total if total != 0 else 0
        return result


class F1Score:
    def __init__(self, overlaps = [0.1, 0.25, 0.5]):
        self.overlaps = overlaps

    def __call__(self, predictions, targets) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        #self.tps = np.zeros((len(self.overlaps), self.classes))
        #self.fps = np.zeros((len(self.overlaps), self.classes))
        #self.fns = np.zeros((len(self.overlaps), self.classes))

        result = {}
        for o in self.overlaps:
            result[f'F1@{int(o*100)}'] = 0.0

        batches_count = predictions.shape[0]
        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())
        for p, t in zip(predictions, targets):

            # Skip all padding
            mask = np.logical_not(np.isin(t, [-100]))
            t = t[mask]
            p = p[mask]

            for i, overlap in enumerate(self.overlaps):
                tp, fp, fn = self.f_score(
                    p.tolist(),
                    t.tolist(),
                    overlap
                )
                
                #self.tps[i] += tp
                #self.fps[i] += fp
                #self.fns[i] += fn

                f1 = self.get_f1_score(tp, fp, fn)
                result[f'F1@{int(overlap*100)}'] += f1

        for o in self.overlaps:
            result[f'F1@{int(o*100)}'] /= batches_count 
        return result

    @staticmethod
    def f_score(predictions, targets, overlap, ignore_classes=[-100]):
        p_label, p_start, p_end = _get_labels_start_end_time(predictions, ignore_classes)
        y_label, y_start, y_end = _get_labels_start_end_time(targets, ignore_classes)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)

    @staticmethod
    def get_f1_score(tp, fp, fn):
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0
        
        if precision + recall != 0.0:
            return 2.0 * (precision * recall) / (precision + recall)
        else:
            return 0.0

#class F1Score2():
#    def __init__(
#        self,
#        overlaps = (0.1, 0.25, 0.5),
#        ignore_ids = [-100],
#        window_size: int = 1,
#        num_classes: int = None,
#    ):
#        super(F1Score2, self).__init__()
#        self.overlaps = overlaps
#        self.ignore_ids = ignore_ids
#        self.num_classes = num_classes
#        self.reset()
#
#    # noinspection PyAttributeOutsideInit
#    def reset(self):
#        if self.num_classes is None:
#            shape = (len(self.overlaps), 1)
#        else:
#            shape = (len(self.overlaps), self.num_classes)
#        self.tp = np.zeros(shape)
#        self.fp = np.zeros(shape)
#        self.fn = np.zeros(shape)
#
#    def get_deque_median(self):
#        medians = {}
#        aggregate_scores = defaultdict(list)
#        for score_dict in self.deque:
#            for n, v in score_dict.items():
#                aggregate_scores[n].append(v)
#        for name, scores in aggregate_scores.items():
#            medians[name] = np.median(scores)
#        return medians
#
#    def add(
#            self,
#            targets,
#            predictions
#    ) -> dict:
#        """
#
#        :param targets: tensor of shape [batch_size, seq_len]
#        :param predictions: tensor of shape [batch_size, seq_len]
#        :return:
#        """
#
#        targets = np.array(targets)
#        predictions = np.array(predictions)
#        for target, pred in zip(targets, predictions):
#            current_result = {}
#            mask = np.logical_not(np.isin(target, self.ignore_ids))
#            target = target[mask]
#            pred = pred[mask]
#
#            for s in range(len(self.overlaps)):
#                tp1, fp1, fn1 = F1Score.f_score(
#                    pred.tolist(),
#                    target.tolist(),
#                    self.overlaps[s]
#                )
#                self.tp[s] += tp1
#                self.fp[s] += fp1
#                self.fn[s] += fn1
#
#                current_f1 = self.get_f1_score(tp1, fp1, fn1)
#                current_result[f"F1@{int(self.overlaps[s]*100)}"] = current_f1
#
#        return current_result
#
#    def summary(self) -> dict:
#        result = {}
#        for s in range(len(self.overlaps)):
#            f1_per_class = self.get_f1_score(tp=self.tp[s], fp=self.fp[s], fn=self.fn[s])
#            result[f"F1@{int(self.overlaps[s]*100)}"] = np.mean(f1_per_class)
#
#        return result
#
#    @staticmethod
#    def get_vectorized_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
#        """
#        Args:
#            tp: [num_classes]
#            fp: [num_classes]
#            fn: [num_classes]
#        Returns:
#            [num_classes]
#        """
#        return 2 * tp / (2 * tp + fp + fn + 0.00001)
#
#    @staticmethod
#    def get_f1_score(tp: float, fp: float, fn: float) -> float:
#        if tp + fp != 0.0:
#            precision = tp / (tp + fp)
#            recall = tp / (tp + fn)
#        else:
#            precision = 0.0
#            recall = 0.0
#
#        if precision + recall != 0.0:
#            f1 = 2.0 * (precision * recall) / (precision + recall)
#        else:
#            f1 = 0.0
#
#        return f1

class EditDistance:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        batch_scores = []
        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())
        for pred, target in zip(predictions, targets):

            # Skip all padding
            mask = np.logical_not(np.isin(target, [-100]))
            target = target[mask]
            pred = pred[mask]

            batch_scores.append(self.edit_score(
                predictions=pred.tolist(),
                targets=target.tolist(),
                norm=self.normalize
            ))

        # Mean in the batch
        return sum(batch_scores) / len(batch_scores)
    
    @staticmethod
    def edit_score(predictions, targets, norm=True, ignore_classes=[-100]):
        P, _, _ = _get_labels_start_end_time(predictions, ignore_classes)
        Y, _, _ = _get_labels_start_end_time(targets, ignore_classes)
        return EditDistance.levenstein(P, Y, norm)
    
    @staticmethod
    def levenstein(p, y, norm=False):
        m_row = len(p) 
        n_col = len(y)
        D = np.zeros([m_row+1, n_col+1], float)
        for i in range(m_row+1):
            D[i, 0] = i
        for i in range(n_col+1):
            D[0, i] = i

        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if y[j-1] == p[i-1]:
                    D[i, j] = D[i-1, j-1]
                else:
                    D[i, j] = min(D[i-1, j] + 1,
                                D[i, j-1] + 1,
                                D[i-1, j-1] + 1)
        
        if norm:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score

#class Edit:
#    def __init__(self, ignore_ids, window_size: int = 1):
#        super(Edit, self).__init__()
#        self.ignore_ids = ignore_ids
#        self.reset()
#
#    def reset(self):
#        self.values = []
#
#    def get_deque_median(self):
#        return np.median(self.deque)
#
#    def add(
#        self, targets, predictions
#    ) -> float:
#        """
#
#        :param targets: torch tensor with shape [batch_size, seq_len]
#        :param predictions: torch tensor with shape [batch_size, seq_len]
#        :return:
#        """
#        targets, predictions = np.array(targets), np.array(predictions)
#        for target, pred in zip(targets, predictions):
#            mask = np.logical_not(np.isin(target, self.ignore_ids))
#            target = target[mask]
#            pred = pred[mask]
#
#            current_score = EditDistance.edit_score(
#                predictions=pred.tolist(),
#                targets=target.tolist(),
#            )
#
#            self.values.append(current_score)
#
#        return current_score
#
#    def summary(self) -> float:
#        if len(self.values) > 0:
#            return np.array(self.values).mean()
#        else:
#            return 0.0

