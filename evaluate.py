from simple_model import LossModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_iou(pred, target, eps=1e-7):
    xmin_pt, ymin_pt, xmax_pt, ymax_pt = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    x1 = torch.max(xmin_pt, xmin_gt)
    y1 = torch.max(ymin_pt, ymin_gt)
    x2 = torch.min(xmax_pt, xmax_gt)
    y2 = torch.min(ymax_pt, ymax_gt)

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_pred = (xmax_pt - xmin_pt) * (ymax_pt - ymin_pt)
    area_gt = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
    union = area_gt + area_pred - inter + eps

    return inter / union


def evaluate(dataset, model_path, iou_thres=0.5):
    model = LossModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    total = 0
    iou_scores = []

    with torch.no_grad():
        for pt, gt,_ in dataset:
            pt = pt.unsqueeze(0).to(device)  # shape: [1, 4]
            gt = gt.unsqueeze(0).to(device)

            out = model(pt)
            iou = compute_iou(out, gt).item()
            iou_scores.append(iou)
            total += 1

            if iou >= iou_thres:
                TP += 1
            else:
                FP += 1
                FN += 1

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = TP / total

    print(f"Total samples: {total}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "mean_iou": sum(iou_scores) / len(iou_scores)
    }
evaluate(validation_set,"od_models/best.pt",iou_thres=0.05)