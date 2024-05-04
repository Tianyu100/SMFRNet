import torch
import torch.nn as nn
import torch.nn.functional as F



class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def crop(self, w, h, target):  #保证当预测图与gt图片大小不一致的时候，把与预测图相同的gt部分截出来
        target_all = target.size()# nt, ht, wt
        offset_w, offset_h = (target_all[2] - w) // 2, (target_all[1] - h) // 2   #确定截取位置
        if offset_w > 0 and offset_h > 0:
            target = target[:, offset_h:-offset_h, offset_w:-offset_w]             #把预测图对应的gt部分截取出来

        return target

    def to_one_hot(self, target, size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target = torch.LongTensor(n, 1, h, w)
        if target.is_cuda:
            ymask = ymask.cuda(target.get_device())
            new_target = new_target.cuda(target.get_device())

        new_target[:, 0, :, :] = torch.clamp(target.detach(), 0, c - 1) #tensor.detach()：不计算张量梯度
       # torch.clamp(input, min, max, out=None) 将输入input张量每个元素的范围限制到区间[min, max]，返回结果到一个新张量。
        ymask.scatter_(1, new_target, 1.0)

        return torch.autograd.Variable(ymask)

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc

        """
        #加入权重weit
        weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)


        gt = torch.squeeze(gt)   #torch.squeeze：将input中大小为1的所有维都删除，返回张量类型

        n, c, h, w = pred.shape
        # log_p = F.log_softmax(pred, dim=1)
        log_p = torch.softmax(pred, dim=1)

        # softmax so that predicted map can be distributed in [0, 1]
        # pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        gt = self.crop(w, h, gt)
        one_hot_gt = self.to_one_hot(gt, log_p.size())

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)
        weit = weit.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext*weit, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b*weit, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        # BF1 = 2 * P * R / (P + R + 1e-7)
        smooth = 1e-7
        BF1 = (2 * P * R) / (P + R + smooth)
        # BF1 = (2 * P * R + smooth) / (P + R + smooth)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss



# for debug
if __name__ == "__main__":
    import torch.optim as optim
    from torchvision.models import segmentation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.randn(8, 3, 200, 200).to(device)
    gt = torch.randint(0,10,(8,224,224)).to(device)  #返回矩阵元素在0~10(10取不到)，然后见https://blog.csdn.net/qq_43332629/article/details/106092700
    # print(gt)  3维矩阵

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)

