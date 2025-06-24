import torch
import torch.nn as nn
import torch.nn.functional as F

#structure mask loss
bce = nn.BCELoss(reduction='mean')

def multi_bce(preds, gt):
    m_loss = bce(preds[3], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        loss += bce(preds[i], gt) * ((2 ** i) / 16)  # loss
        #loss += bce(preds[i], gt) * ((1+i) / 4)
    return loss + m_loss

def single_structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
    

def multi_structure_loss(preds, gt):
    m_loss = single_structure_loss(preds[3], gt)
    loss = 0.
    for i in range(0, len(preds) - 1):
        # loss += bce(preds[i], gt) * ((2 ** i) / 16)  # loss
        loss += (structure_loss(preds[i], gt)) * ((1+i) / 4)
    return loss + m_loss, m_loss
    
def structure_loss(preds, gt):

    loss_P1 = single_structure_loss(preds[0], gt)
    loss_P2 = single_structure_loss(preds[1], gt)
    loss_P3 = single_structure_loss(preds[2], gt)
    loss_P4 = single_structure_loss(preds[3], gt)

    alpha, beta, gamma, zeta = 1.0, 1.5, 1.5, 2.0   
    #alpha, beta, gamma, zeta = 1.0, 1.0, 1.0, 1.0
    loss = alpha * loss_P1 + beta * loss_P2 + gamma * loss_P3 + zeta * loss_P4
    return loss


def edge_loss(pred, gt, threshold=0.5):
    
    # Convert the ground truth (Canny edge map) to a binary mask
    gt_edges = gt.float() / 255.0  # Convert Canny edges from [0, 255] to [0, 1]
    
    # Apply threshold to the predicted edges to create binary predictions
    pred_edges = (pred > threshold).float()  # Binarize the predicted edges

    # Use BCE loss to compare binary edge maps
    return bce(pred_edges, gt_edges)

    
def multi_edge_loss(preds, gt):
   
    e_loss1 = edge_loss(preds[0], gt)
    e_loss2 = edge_loss(preds[1], gt)
    e_loss3 = edge_loss(preds[2], gt)
    e_loss4 = edge_loss(preds[3], gt)
    
    alpha, beta, gamma, zeta = 1.0, 1.5, 1.5, 2.0
    loss = alpha*e_loss1+ beta*e_loss2+ gamma*e_loss3+ zeta*e_loss4
    
    return loss
    
