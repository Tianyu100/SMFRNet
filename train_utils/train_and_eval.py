import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import train_utils.distributed_utils as utils
import mySodMetrics_recoder
import numpy as np
import pytorch_iou,pytorch_ssim
# import boundary_loss
# import lovasz_losses as L


metrics_recoder=mySodMetrics_recoder.MetricRecorder()

def criterion(inputs, target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt=torch.clamp(predict, 0.001, 0.999)
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
    
def dice_loss(pred,target,ep=1e-8):
    intersection = 2 * torch.sum(pred * target) + ep
    union = torch.sum(pred) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
# boundary_loss = boundary_loss.BoundaryLoss()


def bce_ssim_loss(pred, target):
    weit  = 1+5*torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)-target)
    wbce  = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
#     wbce=bce_loss(pred, target)
    wbce_out = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    
    ssim_out = 1 - ssim_loss(pred, target)
    
    inter = ((pred*target)*weit).sum(dim=(2,3))
    union = ((pred+target)*weit).sum(dim=(2,3))
    wiou_out  = 1-(inter+1)/(union-inter+1)

#     boundary_out = boundary_loss(pred, target)

    Focal_loss=BCEFocalLoss()
    focal_out = Focal_loss.forward(pred, target)

#     pred=torch.squeeze(pred)
#     mask=torch.squeeze(target)
#     lovasz_loss = L.lovasz_hinge(pred,target)
    
    loss = wbce_out + ssim_out + wiou_out+ focal_out
#     loss = wbce_out + ssim_out + wiou_out

    return loss.mean()


def muti_bce_loss_fusion(d0, d1, d2, d3, d4,d5,d6,labels_v):
    loss0 = bce_ssim_loss(d0, labels_v)
    loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4+loss5+loss6  # + 5.0*lossa
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
    return loss


def evaluate(model, data_loader, device):
    model.eval()
    mae_metric = metrics_recoder.mae
    f1_metric = metrics_recoder.fm
    sm_metric = metrics_recoder.sm

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            pred = model(images)   #tensor:[1,1,h,w]

            # pred=torch.squeeze(output)     #[h,w]
            pred=torch.squeeze(pred).cpu().numpy()
            targets=torch.squeeze(targets).cpu().numpy()
            output,targets=np.float16(pred),np.float16(targets)

            metrics_recoder.update(pre=output, gt=targets)

    sm =sm_metric.get_results()["sm"]
    mae=mae_metric.get_results()["mae"]
    fm =f1_metric.get_results()["fm"]
    f1mean=fm["curve"].mean()


    return mae, f1mean,sm

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target= image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(image)
#             loss = muti_bce_loss_fusion(out[0],out[1],out[2],out[3],out[4],out[5],out[6],target)
            loss = criterion(out)

            # print("loss:%3f\n", loss.item())

        optimizer.zero_grad()
        if scaler is not None:
#             print("use amp..............................")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
#             print("unuse amp..............................")
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group




