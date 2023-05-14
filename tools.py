from collections import defaultdict, deque
import datetime
import math
import sys
import time
import torch
import torch.distributed as dist
import torchvision
import numpy as np
from tqdm import tqdm


def collate_fn(batch):
    """Функция для преобразования целевых переменных в кортеж"""
    return tuple(zip(*batch))

def compute_total_iou(pred_boxes, gt_boxes):
    """ Вычисляет IoU для тензоров bounding boxes """
    iou = torchvision.ops.box_iou(pred_boxes, gt_boxes)
    return iou.diag()

def compute_iou(bbox1, bbox2):
    """
    IoU для двух bboxes
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Площадь пересечения bboxes
    x_intersect = max(0, min(x2, x4) - max(x1, x3))
    y_intersect = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_intersect * y_intersect

    #Общая площадь bboxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Вычисление IoU
    iou = intersection_area / union_area

    return iou    

def rescale_boxes(annotations, original_size, new_size):
    """Функция для пропорционального изменения координат bounding boxes"""
    w_ratio = new_size[0] / original_size[0]
    h_ratio = new_size[1] / original_size[1]
    rescaled_annotations = []

    for annotation in annotations:
        rescaled_annotation = annotation.copy()
        bbox = annotation['bbox']
        x, y, w, h = bbox
        x_rescaled = x * w_ratio
        y_rescaled = y * h_ratio
        w_rescaled = w * w_ratio
        h_rescaled = h * h_ratio
        rescaled_bbox = [x_rescaled, y_rescaled, w_rescaled, h_rescaled]
        rescaled_annotation['bbox'] = rescaled_bbox
        rescaled_annotations.append(rescaled_annotation)
        
    return rescaled_annotations

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.7):
    """
    Вычисление метрик Accurary, и Precision
    """
    num_preds = len(pred_boxes)
    num_true = len(true_boxes)

    # Если не с чем сверять, возвращаем 0
    if num_true == 0:
        return 0, 0, 0

    # Сортировка предсказанных меток
    sorted_indices = torch.argsort(pred_scores, descending=True)
    sorted_pred_boxes = pred_boxes[sorted_indices]
    sorted_pred_labels = pred_labels[sorted_indices]

    true_positives = np.zeros(num_preds)
    false_positives = np.zeros(num_preds)
    false_negatives = np.zeros(num_true)

    # Каждый объект сравнивается с Grount truth boxes
    for i in range(num_preds):
        box = sorted_pred_boxes[i]
        
        # вычисление IoU со всеми ground truth boxes
        ious = torch.stack([compute_iou(box, true_boxes[j]) for j in range(num_true)])
        best_match_index = torch.argmax(ious)

        # Если IoU больше параметра treshhold, отмечаем его как True Positive
        if ious[best_match_index] > iou_threshold and false_negatives[best_match_index] == 0:
            true_positives[i] = 1
            false_negatives[best_match_index] = 1
        else:
            false_positives[i] = 1

    accuracy = np.sum(true_positives) / num_preds
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    # recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    return accuracy, precision

def evaluate_data(model, data_loader, device): 
    """Функция оценки предсказания, выводит метрики точности для набора данных """
    model.eval()
    # metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('mean_iou', SmoothedValue(fmt='{value:.4f}'))
    # metric_logger.add_meter('mean_accuracy', SmoothedValue(fmt='{value:.4f}'))
    # metric_logger.add_meter('mean_presicion', SmoothedValue(fmt='{value:.4f}'))
    header = 'Test:'

    total_iou = 0.0
    total_acc = 0.0
    total_prec = 0.0
    # total_rec = 0.0
    total_objects = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            # оценка метрик для каждого изображения
            for i, target in enumerate(targets):
                gt_boxes = target['boxes'].cpu()
                pred_boxes = outputs[i]['boxes'].cpu()
                iou = compute_total_iou(pred_boxes, gt_boxes).sum().item()
                acc, prec = calculate_metrics(pred_boxes, outputs[i]['labels'].cpu(), outputs[i]['scores'].cpu(), gt_boxes, target['boxes'].cpu())
                total_iou += iou
                total_acc +=  acc
                total_prec += prec
                # total_rec += rec
                total_objects += 1

    mean_iou = total_iou / total_objects
    mean_acc = total_acc / total_objects
    mean_prec = total_prec / total_objects
    # mean_rec = total_rec / total_objects
    print(f"\nMean IoU: {mean_iou:.4f}, Mean Accuracy: {mean_acc:.4f}, Mean Precision: {mean_prec:.4f}")
    # metric_logger.update(mean_iou=mean_iou)
    # metric_logger.update(mean_accuracy=mean_acc)
    # metric_logger.update(mean_precision=mean_prec)

    return mean_iou, mean_acc, mean_prec
    # return metric_logger

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

