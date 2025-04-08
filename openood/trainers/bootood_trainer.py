import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openood.utils import Config
import openood.utils.comm as comm
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random
from openood.utils import Config


class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        label_one_hot = F.one_hot(torch.clamp(targets, min=0), num_classes=num_classes).float()
        label_one_hot = torch.where(targets.unsqueeze(-1) >= 0, label_one_hot, 1/num_classes)
        return F.cross_entropy(logits, label_one_hot, reduction=self.reduction)


class CustomCrossEntropyNormLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.muG = None

    def update_muG(self, global_mean):
        self.muG = global_mean

    def forward(self, logits, targets):
        if self.muG is None:
            self.muG = torch.zeros(logits.size(1), device=logits.device)

        negative_mask = targets < 0
        ce_loss = F.cross_entropy(logits, targets, reduction='sum', ignore_index=-1)
        norm_loss = F.mse_loss(logits * negative_mask.unsqueeze(1), self.muG.expand_as(logits), reduction='sum')
        total_loss = ce_loss + norm_loss

        if self.reduction == 'mean':
            return total_loss / len(targets)
        elif self.reduction == 'sum':
            return total_loss
        else:
            raise ValueError('Invalid reduction type.')


def evaluate_model(model, test_loader, criterion, test=False):
    model.eval()
    loss, correct, total = 0.0, 0, 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        batch_loss = criterion(output, target)
        loss += batch_loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    acc = correct / total
    loss /= total
    print(f"{'Test' if test else 'Val'}/loss: {loss:.4f}\tAccuracy: {acc:.4f}")
    return loss


class BootoodTrainer:
    def __init__(self, net, train_loader, config: Config):
        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.optimizer.num_epochs,
            eta_min=1e-6,
        )

        loss_name = getattr(config, 'loss_name', 'CustomCrossEntropyLoss')
        if loss_name == 'CustomCrossEntropyLoss':
            self.criterion = CustomCrossEntropyLoss()
        elif loss_name == 'CustomCrossEntropyNormLoss':
            self.criterion = CustomCrossEntropyNormLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.net.to(self.device)
        self.net.train()
        self.net.fc.register_forward_hook(hook)

    def train_epoch(self, epoch_idx):
        self.net.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch_idx}"):
            data, target = batch['data'].to(self.device), batch['label'].to(self.device)
            logits = self.net(data)
            loss = self.criterion(logits, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += data.size(0)

        self.scheduler.step()

        loss_avg = total_loss / total
        acc = correct / total

        if epoch_idx == self.config.optimizer.num_epochs:
            torch.save(self.net.state_dict(), './results/mymodel_ckpt.pth')

        metrics = {
            'epoch_idx': epoch_idx,
            'loss': self.save_metrics(loss_avg),
            'acc': acc,
        }
        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        return np.mean([x for x in all_loss])

    def evaluate(self, test_loader):
        self.net.eval()
        with torch.no_grad():
            loss = evaluate_model(self.net, test_loader, self.criterion, test=True)
        return loss