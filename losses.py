def loss_MAE(out,gt):
    return torch.mean(torch.abs(out - gt))

def hybrid_loss(pred, target, epoch, switch_epoch=20):
    if epoch < switch_epoch:
        return loss_MAE(pred,target)  # MAE
    else:
        return iou_loss(pred, target)  # your IoU loss
    

# import numpy as np
def iou_loss(pred, target, eps=1e-7):
    # Convert to [x1, y1, x2, y2]
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_gt = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_gt - inter + eps

    iou = inter / union
    return 1 - iou.mean()

def angle_cost(pred, target, eps=1e-7):
    # Compute box centers
    cx_pred = (pred[:, 0] + pred[:, 2]) / 2
    cy_pred = (pred[:, 1] + pred[:, 3]) / 2
    cx_gt = (target[:, 0] + target[:, 2]) / 2
    cy_gt = (target[:, 1] + target[:, 3]) / 2

    # Center offset
    s_cw = cx_gt - cx_pred
    s_ch = cy_gt - cy_pred
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + eps)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = np.sqrt(2) / 2

    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

    angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - torch.pi / 4), 2)
    return angle_cost.mean()

def angle_and_distance_cost(pred, target, eps=1e-7):
    cx_pred = (pred[:, 0] + pred[:, 2]) / 2
    cy_pred = (pred[:, 1] + pred[:, 3]) / 2
    cx_gt = (target[:, 0] + target[:, 2]) / 2
    cy_gt = (target[:, 1] + target[:, 3]) / 2

    s_cw = cx_gt - cx_pred
    s_ch = cy_gt - cy_pred
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + eps)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = np.sqrt(2) / 2

    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - torch.pi / 4), 2)

    cw = torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])
    ch = torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1])

    rho_x = (s_cw / (cw + eps)) ** 2
    rho_y = (s_ch / (ch + eps)) ** 2
    gamma = 2 - angle_cost

    # Clamp exp input for stability
    exp1 = torch.exp(-gamma * rho_x.clamp(max=10))  # avoid overflow
    exp2 = torch.exp(-gamma * rho_y.clamp(max=10))

    distance_cost = 2 - exp1 - exp2
    distance_cost = torch.clamp(distance_cost, min=0.0)  # final safety clamp

    return angle_cost.mean(), distance_cost.mean()


def shape_cost(pred, target, eps=1e-7):
    # Compute box centers
    x1_pred , y1_pred= pred[:, 0],pred[:, 1]
    x2_pred , y2_pred = pred[:, 2],pred[:, 3]
    x1_gt , y1_gt=  target[:, 0],target[:, 1]
    x2_gt , y2_gt = target[:, 2],target[:, 3]

    w_gt = x2_gt - x1_gt
    w_pt = x2_pred - x1_pred

    h_gt = y2_gt - y1_gt
    h_pt = y2_pred - y1_pred

    omega_w  = torch.abs(w_pt - w_gt) / (torch.max(w_pt, w_gt) + eps)
    omega_h = torch.abs(h_pt - h_gt) / (torch.max(h_pt, h_gt) + eps)

    shape = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
    return shape.mean()

def hybrid_loss_sum(pred, target , epoch, switch_epoch=50): #alpha beta param loss
    lossm = loss_MAE(pred,target)  # MAE
    lossi = iou_loss(pred, target)  # your IoU loss
    lossa = angle_cost(pred, target)
    if epoch < switch_epoch:
        return lossm 
    else:
        return Siou_loss_full(pred,target)
def Siou_loss_full(pred, target, eps=1e-7, return_components=False):
    angle, dist = angle_and_distance_cost(pred, target)
    shape = shape_cost(pred, target)
    iou_term = iou_loss(pred, target)  # already returns (1 - IoU)

    # Normalization to balance contribution
    angle_norm = angle / 2.0
    dist_norm = dist / 2.0
    shape_norm = shape * 5.0

    directional_term = (dist_norm +  shape_norm) / 2.0
    total_loss = iou_term + directional_term
    # total_loss =  directional_term


    if return_components:
        return total_loss, {
            "iou": iou_term.item(),
            "angle": angle.item(),
            "angle_norm": angle_norm.item(),
            "distance": dist.item(),
            "distance_norm": dist_norm.item(),
            "shape": shape.item(),
            "shape_norm": shape_norm.item()
        }
    else:
        return total_loss


