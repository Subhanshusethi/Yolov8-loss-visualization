from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import os
from torch.utils.data import random_split
from tqdm import tqdm
from collections import defaultdict



class LossData(Dataset):
    def __init__(self, gt_path, purturbed_path):
        super().__init__()
        self.gt = []
        self.pt = []
        self.filenames = []

        gt_files = sorted(os.listdir(gt_path))
        pt_files = sorted(os.listdir(purturbed_path))

        for gt_file, pt_file in zip(gt_files, pt_files):
            with open(os.path.join(gt_path, gt_file)) as gt:
                gt_lines = gt.readlines()
            with open(os.path.join(purturbed_path, pt_file)) as pt:
                pt_lines = pt.readlines()

            for line_num, (gt_line, pt_line) in enumerate(zip(gt_lines, pt_lines)):
                self.gt.append(gt_line.strip().split())
                self.pt.append(pt_line.strip().split())
                self.filenames.append(f"{gt_file}_{line_num}")

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        gt_values = self.gt[index]
        pt_values = self.pt[index]
        filename = self.filenames[index]

        # Ground truth
        x_gt = float(gt_values[1])
        y_gt = float(gt_values[2])
        w_gt = float(gt_values[3])
        h_gt = float(gt_values[4])

        x1_gt = x_gt - w_gt / 2
        y1_gt = y_gt - h_gt / 2
        x2_gt = x_gt + w_gt / 2
        y2_gt = y_gt + h_gt / 2

        # Perturbed
        x_pt = float(pt_values[1])
        y_pt = float(pt_values[2])
        w_pt = float(pt_values[3])
        h_pt = float(pt_values[4])

        x1_pt = x_pt - w_pt / 2
        y1_pt = y_pt - h_pt / 2
        x2_pt = x_pt + w_pt / 2
        y2_pt = y_pt + h_pt / 2

        gt = torch.tensor([x1_gt, y1_gt, x2_gt, y2_gt], dtype=torch.float32)
        pt = torch.tensor([x1_pt, y1_pt, x2_pt, y2_pt], dtype=torch.float32)

        return pt, gt, filename
