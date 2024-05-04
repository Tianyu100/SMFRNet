# import math
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import train_utils.distributed_utils as utils
# import mySodMetrics_recoder
# import numpy as np
# import pytorch_iou,pytorch_ssim
# from new_boundary_loss import BoundaryLoss
# import lovasz_losses as L
#
# import cv2
#
#
#
# metrics_recoder=mySodMetrics_recoder.MetricRecorder()
#
# def criterion(inputs, target):
#     losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
#     total_loss = sum(losses)
#
#     return total_loss
#
#
# # def evaluate(model, data_loader, device):
# #     model.eval()
# #     mae_metric = utils.MeanAbsoluteError()
# #     f1_metric = utils.F1Score()
# #     # S_metric = sod_metrics.Smeasure()
# #
# #     metric_logger = utils.MetricLogger(delimiter="  ")
# #     header = 'Test:'
# #     with torch.no_grad():
# #         for images, targets in metric_logger.log_every(data_loader, 100, header):
# #             images, targets = images.to(device), targets.to(device)
# #             output = model(images)
# #
# #             # post norm
# #             # ma = torch.max(output)
# #             # mi = torch.min(output)
# #             # output = (output - mi) / (ma - mi)
# #
# #             mae_metric.update(output, targets)
# #             f1_metric.update(output, targets)
# #             # S_metric.step(output, targets)
# #
# #         mae_metric.gather_from_all_processes()
# #         f1_metric.reduce_from_all_processes()
# #
# #     return mae_metric, f1_metric
# class BCEFocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
#         super(BCEFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         pt = torch.sigmoid(predict)  # sigmoide获取概率
#         # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
#         loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
#                     1 - target) * torch.log(1 - pt)
#
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         return loss
#
# def dice_loss(pred,target,ep=1e-8):
#     intersection = 2 * torch.sum(pred * target) + ep
#     union = torch.sum(pred) + torch.sum(target) + ep
#     loss = 1 - intersection / union
#     return loss
#
# # ------- 1. define loss function --------
#
# bce_loss = nn.BCELoss(size_average=True)
# ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
# iou_loss = pytorch_iou.IOU(size_average=True)
# # boundary_loss = boundary_loss.BoundaryLoss()
# def bce_ssim_loss(pred, target):
#
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
#     #     wbce  = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
#     wbce = bce_loss(pred, target)
#     wbce_out = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     #     Focal_loss=BCEFocalLoss()
#     #     focal_out = Focal_loss.forward(pred, target)
#     #     print(focal_out)
#
#     ssim_out = 1 - ssim_loss(pred, target)
#
#     inter = ((pred * target) * weit).sum(dim=(2, 3))
#     union = ((pred + target) * weit).sum(dim=(2, 3))
#     wiou_out = 1 - (inter + 1) / (union - inter + 1)
#
#     boundary_loss = BoundaryLoss()
#     boundary_out=boundary_loss.forward(pred, target)
#
#
#     loss=wbce_out+ssim_out+wiou_out+boundary_out
#
#     return loss
#
# def muti_bce_loss_fusion(d0, d1, d2,d3,labels_v):
#     loss0 = bce_ssim_loss(d0, labels_v)
#     loss1 = bce_ssim_loss(d1, labels_v)
#     loss2 = bce_ssim_loss(d2, labels_v)
#     loss3 = bce_ssim_loss(d3, labels_v)
#     # # 使用正则化
#     # filler = torch.ones_like(labels_v)
#     # att_loss = F.binary_cross_entropy_with_logits(d0,torch.where(d1.max(1)[0] > 0.8,labels_v, filler))
#
#     # iou0 = iou_loss(d0,labels_v)
#     # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
#     loss = loss0 + loss1 + loss2+loss3   # + 5.0*lossa
#     # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
#     # loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
#     # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))
#
#     return loss0, loss
#
#
# def structure_loss(pred, mask):
#     weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
#     wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
#
#     pred=torch.squeeze(pred)
#     mask=torch.squeeze(mask)
#
#     lovasz_loss = L.lovasz_hinge(pred,mask)
#     # pred  = torch.sigmoid(pred)
#     # inter = ((pred*mask)*weit).sum(dim=(2,3))
#     # union = ((pred+mask)*weit).sum(dim=(2,3))
#     # wiou  = 1-(inter+1)/(union-inter+1)
#     return (wbce+lovasz_loss).mean()         #iou位置信息的准确程度,增加IOU权重是否能把小目标的边界对齐呢？
#
#
#
#
# # GitHub repo: https://github.com/bowenc0221/boundary-iou-api
# # Reference: https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
# # General util function to get the boundary of a binary mask.
# # 该函数用于获取二进制 mask 的边界
# def mask_to_boundary(mask, dilation_ratio=0.05):
#     """
#     Convert binary mask to boundary mask.
#     :param mask (numpy array, uint8): binary mask
#     :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
#     :return: boundary mask (numpy array)
#     """
#     mask = mask.astype(np.uint8)
#
#     h, w = mask.shape
#     img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
#     dilation = int(round(dilation_ratio * img_diag))
#     if dilation < 1:
#         dilation = 1
#     # Pad image so mask truncated by the image border is also considered as boundary.
#     new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
#     kernel = np.ones((3, 3), dtype=np.uint8)
#     new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
#
#     # 因为之前向四周填充了0, 故而这里不再需要四周
#     mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
#     # G_d intersects G in the paper.
#     return mask - mask_erode
# def boundary_iou(gt, dt, dilation_ratio=0.005, cls_num=1):
#     """
#     Compute boundary iou between two binary masks.
#     :param gt (numpy array, uint8): binary mask
#     :param dt (numpy array, uint8): binary mask
#     :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
#     :return: boundary iou (float)
#     """
#     gt = gt[0,0]
#     dt = dt[0]
#     gt = gt.cpu().detach().numpy().astype(np.uint8)
#     dt = dt.cpu().detach().numpy().astype(np.uint8)
#
#     boundary_iou_list = []
#     for i in range(cls_num):
#
#         gt_i = (gt == (i+1))
#         dt_i = (dt == (i+1))
#
#         gt_boundary = mask_to_boundary(gt_i, dilation_ratio)
#         dt_boundary = mask_to_boundary(dt_i, dilation_ratio)
#         intersection = ((gt_boundary * dt_boundary) > 0).sum()
#         union = ((gt_boundary + dt_boundary) > 0).sum()
#         if union < 1:
#              boundary_iou_list.append(0)
#         continue
#     boundary_iou = intersection / union
#     boundary_iou_list.append(boundary_iou)
#
#     return np.array(boundary_iou_list)
#
#
# class CEL(nn.Module):
#     def __init__(self):
#         super(CEL, self).__init__()
#         # print("You are using `CEL`!")
#         self.eps = 1e-6
#
#     def forward(self, pred, target):
#         pred = pred.sigmoid()
#         intersection = pred * target
#         numerator = (pred - intersection).sum() + (target - intersection).sum()
#         denominator = pred.sum() + target.sum()
#         return numerator / (denominator + self.eps)
#
# def MINet_loss(pred,mask):
#     bce=F.binary_cross_entropy_with_logits(pred,mask)
#     cel_loss=CEL()
#     cel=cel_loss(pred,mask)
#     loss=bce+cel
#     return loss
#
#
# def evaluate(model, data_loader, device):
#     model.eval()
#     mae_metric = metrics_recoder.mae
#     f1_metric = metrics_recoder.fm
#     sm_metric = metrics_recoder.sm
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Val:'
#     with torch.no_grad():
#         for images, targets in metric_logger.log_every(data_loader, 100, header):
#             images, targets = images.to(device), targets.to(device)
#             pred = model(images)   #tensor:[1,1,h,w]
#
#             # pred=torch.squeeze(output)     #[h,w]
#             pred=torch.squeeze(pred).cpu().numpy()
#             targets=torch.squeeze(targets).cpu().numpy()
#             output,targets=np.float16(pred),np.float16(targets)
#
#             metrics_recoder.update(pre=output, gt=targets)
#
#     sm =sm_metric.get_results()["sm"]
#     mae=mae_metric.get_results()["mae"]
#     fm =f1_metric.get_results()["fm"]
#     f1mean=fm["curve"].mean()
#
#     return mae, f1mean,sm
#
# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#
#     for image, target in metric_logger.log_every(data_loader, print_freq, header):
#         image, target= image.to(device), target.to(device)
#
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             out = model(image)
#             loss0,loss = muti_bce_loss_fusion(out[0],out[1],out[2],out[3],target)
#             # loss=MINet_loss(out,target)
#
#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
#
#         lr_scheduler.step()
#
#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(loss=loss.item(), lr=lr)
#
#     return metric_logger.meters["loss"].global_avg, lr
#
#
#
# def create_lr_scheduler(optimizer,
#                         num_step: int,
#                         epochs: int,
#                         warmup=True,
#                         warmup_epochs=1,
#                         warmup_factor=1e-3,
#                         end_factor=1e-6):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0
#
#     def f(x):
#         """
#         根据step数返回一个学习率倍率因子，
#         注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
#         """
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             # warmup后lr倍率因子从1 -> end_factor
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
#
#
# def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
#     params_group = [{"params": [], "weight_decay": 0.},  # no decay
#                     {"params": [], "weight_decay": weight_decay}]  # with decay
#
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#
#         if len(param.shape) == 1 or name.endswith(".bias"):
#             # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
#             params_group[0]["params"].append(param)  # no decay
#         else:
#             params_group[1]["params"].append(param)  # with decay
#
#     return params_group
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import train_utils.distributed_utils as utils
from torch.autograd import Variable
import mySodMetrics_recoder
import numpy as np
import pytorch_iou, pytorch_ssim
# import boundary_loss
from new_boundary_loss import BoundaryLoss
import lovasz_losses as L

metrics_recoder = mySodMetrics_recoder.MetricRecorder()


def criterion(inputs, target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.3, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        #         pt = torch.sigmoid(predict)  # 两次使用sigmoid，数值不是在[0,1]之间
        pt = torch.clamp(predict, 0.001, 0.999)  # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量

        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
                1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def dice_loss(pred, target, ep=1e-8):
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
    weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    #     wbce  = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
    wbce = bce_loss(pred, target)
    wbce_out = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    Focal_loss = BCEFocalLoss()
    focal_out = Focal_loss.forward(pred, target)
    #     print(focal_out)

    ssim_out = 1 - ssim_loss(pred, target)

    inter = ((pred * target) * weit).sum(dim=(2, 3))
    union = ((pred + target) * weit).sum(dim=(2, 3))
    wiou_out = 1 - (inter + 1) / (union - inter + 1)

    #     boundary_out = boundary_loss(pred, target)
    # boundary_loss = BoundaryLoss()
    # boundary_out = boundary_loss.forward(pred, target)

    pred=torch.squeeze(pred)
    mask=torch.squeeze(target)
    lovasz_loss = L.lovasz_hinge(pred,target)

    #     loss = wbce_out + ssim_out + wiou_out + focal_out+ lovasz_loss
    loss = wbce_out + ssim_out + wiou_out + focal_out+lovasz_loss

    return loss.mean()


def muti_bce_loss_fusion(d0, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_ssim_loss(d0, labels_v)
    # loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0  + loss2 + loss3 + loss4 + loss5 + loss6  # + 5.0*lossa
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
        for images, targets,edges in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            pred = model(images)  # tensor:[1,1,h,w]

            # pred=torch.squeeze(output)     #[h,w]
            pred = torch.squeeze(pred).cpu().numpy()
            targets = torch.squeeze(targets).cpu().numpy()
            output, targets = np.float16(pred), np.float16(targets)

            metrics_recoder.update(pre=output, gt=targets)

    sm = sm_metric.get_results()["sm"]
    mae = mae_metric.get_results()["mae"]
    fm = f1_metric.get_results()["fm"]
    f1mean = fm["curve"].mean()

    return mae, f1mean, sm


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        # print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)


def MINet_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    cel_loss = CEL()
    cel = cel_loss(pred, mask)
    loss = bce + cel
    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image,target,edge in metric_logger.log_every(data_loader, print_freq, header):

        image = image.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        edge = edge.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            image, target,edge = Variable(image.cuda(), requires_grad=False), Variable(target.cuda(), requires_grad=False),            Variable(edge.cuda(), requires_grad=False)

        else:
            image, target = Variable(image, requires_grad=False), Variable(target, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = model(image)
        loss0 = muti_bce_loss_fusion(out[1],out[2], out[3], out[4], out[5], out[6], target)
        loss1=bce_loss(out[0],edge)
        loss=loss0+loss1

        #         loss= MINet_loss(out,target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        #         image, target= image.to(device), target.to(device)
        #         with torch.cuda.amp.autocast(enabled=scaler is not None):
        #             optimizer.zero_grad()
        #             out = model(image)
        #             loss = muti_bce_loss_fusion(out[0],out[1],out[2],out[3],out[4],out[5],out[6],target)
        # #             loss= MINet_loss(out,target)

        #              print("loss:%3f\n", loss.item())

        #         if scaler is not None:
        #             scaler.scale(loss).backward()

        #             lr_scheduler.step()

        #             scaler.step(optimizer)
        #             scaler.update()
        #         else:
        #             loss.backward()
        #             optimizer.step()
        # #         lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        del loss

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




