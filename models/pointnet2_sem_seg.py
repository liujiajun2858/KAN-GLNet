import torch.nn as nn
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionKPconv, \
#     PointNetSetAbstractionAttention
from models.pointnet2_utils import *
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation, \
    PointNetSetAbstractionMsgAttention


from block.normal import ContraNorm

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        # self.sa1 = PointNetSetAbstraction(1024, 0.01, 32, 9 + 3, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction(256, 0.02, 32, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(64, 0.04, 32, 128 + 3, [128, 128, 256], False)
        #self.sa4 = PointNetSetAbstraction(16, 0.08, 32, 256 + 3, [256, 256, 512], False)
         #PointNetSetAbstractionAttention1
        self.sa1 = PointNetSetAbstractionAttention1(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstractionAttention1(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstractionAttention1(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstractionAttention1(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])


        self.conv1 = nn.Conv1d(128, 128, 1)
        #self.bn1 = nn.BatchNorm1d(128)
        self.bn1 = ContraNorm(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, kernel_size=1, padding=0)
        # self.conv1 = FastKANConv1DLayer(128, 128, kernel_size=1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = FastKANConv1DLayer(128, num_classes, kernel_size=1)



    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)


        l0_points_change=self.conv1(l0_points)


        l0_points_change = l0_points_change.permute(0, 2, 1) #添加的，只需要改这里
        l0_points_bn=self.bn1(l0_points_change)
        l0_points_bn=l0_points_bn.permute(0, 2, 1)#添加的，只需要改这里


        x = self.drop1(F.relu(l0_points_bn))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points

        # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)

        #return x, l4_points
      #x：(batch_size，npoints,类别数（2）)
    #l4——points第一个维度：批次中的样本数量。
              # 第二个维度：每个点的特征维度数。
              # 第三个维度：经过 sa4 层后的点数量。




class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

##Varifocal Loss损失函数   Varifocal_loss
class Varifocal_loss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(Varifocal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target, trans_feat, weight):
        # 将 target 转换为 one-hot 编码格式
        target_one_hot = F.one_hot(target, num_classes=pred_logits.shape[-1]).float()

        # 计算预测概率
        pred_probs = torch.sigmoid(pred_logits)

        # 计算 Focal Loss 中的 modulating factor
        focal_weight = (1 - pred_probs) ** self.gamma

        # 计算 BCE Loss，这里 target 需要是 one-hot 编码的形式
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_one_hot, weight=weight, reduction='none')

        # 对正样本额外增加一个误差平方项
        alpha_factor = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        varifocal_term = torch.where(target_one_hot == 1, (1 - pred_probs).pow(2),
                                     torch.tensor(0., device=pred_logits.device))
        loss = alpha_factor * (focal_weight * bce_loss + varifocal_term)

        # 返回损失的均值
        return loss.mean()


# #按一定比例混合似然和va函数
# class get_loss(nn.Module):
#     def __init__(self,  gamma=2.0, alpha=0.25,l2_lambda=0.1): #原alpha是0.75，改为0.25损失降了不少
#         super(get_loss, self).__init__()
#
#         self.siran_loss = siran_loss()  # 需要传递参数
#         self.varifocal_loss = Varifocal_loss(gamma, alpha)  # 需要传递参数
#
#     def forward(self, pred, target, trans_feat, weight):
#         loss_siran = self.siran_loss(pred, target, trans_feat, weight)
#         loss_varifocal = self.varifocal_loss(pred, target, trans_feat, weight)
#
#         # 组合两
#         # 个损失函数
#         u = 0.1
#         combined_loss = u * loss_siran + (1 - u) * loss_varifocal
#
#         return combined_loss






#PaCoLoss



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# #128
# class get_loss(nn.Module):
#     def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=4, num_classes=2, use_fp16=False):
#         super(get_loss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = temperature if base_temperature is None else base_temperature
#         self.K = K
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.supt = supt
#         self.num_classes = num_classes
#         self.use_fp16 = use_fp16
#
#     def forward(self, pred, target, trans_feat=None, weight=None):
#         device = torch.device('cuda' if pred.is_cuda else 'cpu')
#
#         # Assuming pred is the feature matrix
#         features = pred
#         labels = target.contiguous().view(-1, 1)
#         batch_size = features.shape[0]
#
#         if self.use_fp16:
#             features = features.half()
#
#         # Mask generation
#         mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
#
#         # Compute dot product in chunks to save memory
#         anchor_dot_contrast = torch.zeros(batch_size, batch_size, device=device, dtype=features.dtype)
#         chunk_size = 1024  # You can adjust this based on your memory constraints
#         for i in range(0, batch_size, chunk_size):
#             for j in range(0, batch_size, chunk_size):
#                 anchor_dot_contrast[i:i+chunk_size, j:j+chunk_size] = torch.div(
#                     torch.matmul(features[i:i+chunk_size], features[j:j+chunk_size].T),
#                     self.temperature
#                 )
#
#         # Numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # Mask out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).view(-1, 1).to(device),
#             0
#         )
#
#         mask = mask * logits_mask
#
#         # Add ground truth
#         one_hot_label = F.one_hot(labels[:batch_size].view(-1,), num_classes=self.num_classes).to(torch.float32)
#         mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)
#
#         # Compute log_prob
#         logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
#
#         # Compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#         # Compute loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.mean()
#
#         if weight is not None:
#             loss = loss * weight
#
#         return loss

# Example usage:
# criterion = get_loss(alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=2, use_fp16=True)









if __name__ == '__main__':
    import  torch
    model = get_model(2) #之前是13
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))







