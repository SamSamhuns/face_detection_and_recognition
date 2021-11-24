import torch
import numpy as np


def accuracy(output: torch.tensor, target: torch.tensor):
    """
    Vanilla accuracy: TP + TN / (TP + TN + FP + FN)
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def accuracy_mse(output: torch.tensor, target: torch.tensor):
    """
    Accuracy when using regression rather than classification. Just ignore it.
    """
    with torch.no_grad():
        assert len(output) == len(target)
        correct = 0
        correct += torch.sum(((output - target).abs() < 1)).item()
    return correct / len(target)


def acc_relaxed(output: torch.tensor, target: torch.tensor):
    """
    Function so that 101 age classes correspond to 8 age classes,
    for Adience dataset. Turns out this results in the same value as the vanilla
    accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        correct = 0
        for p, t in zip(pred, target):
            if (0 <= p < 3) and (0 <= t < 3):
                correct += 1
            elif (3 <= p < 7) and (3 <= t < 7):
                correct += 1
            elif (7 <= p < 13.5) and (7 <= t < 13.5):
                correct += 1
            elif (13.5 <= p < 22.5) and (13.5 <= t < 22.5):
                correct += 1
            elif (22.5 <= p < 35) and (22.5 <= t < 35):
                correct += 1
            elif (35 <= p < 45.5) and (35 <= t < 45.5):
                correct += 1
            elif (45.5 <= p < 56.5) and (45.5 <= t < 56.5):
                correct += 1
            elif (56.5 <= p <= 100) and (56.5 <= t <= 100):
                correct += 1
            else:
                pass
    return correct / len(target)


def acc_per_class(output: torch.tensor, target: torch.tensor, num_classes: int):
    """
    Vanilla accuracy per class: TP + TN / (TP + TN + FP + FN)
    Only supported when testing and not supported during training
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        cls_total_count = np.zeros([num_classes])
        cls_correct_count = np.zeros([num_classes])
        for p, t in zip(pred, target):
            cls_total_count[p.item()] += 1
            cls_correct_count[t.item()] += 1 if p == t else 0
        cls_total_count = np.where(cls_total_count == 0., 1., cls_total_count)
        cls_correct_perc = [cc / cls_total_count[i] for i, cc in enumerate(cls_correct_count)]
    return np.asarray(cls_correct_perc)


def top_k_acc(output: torch.tensor, target: torch.tensor, k: int = 3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def top_2_acc(output: torch.tensor, target: torch.tensor):
    return top_k_acc(output, target, k=2)


def top_3_acc(output: torch.tensor, target: torch.tensor):
    return top_k_acc(output, target, k=3)


def top_4_acc(output: torch.tensor, target: torch.tensor):
    return top_k_acc(output, target, k=4)


def top_5_acc(output: torch.tensor, target: torch.tensor):
    return top_k_acc(output, target, k=5)
