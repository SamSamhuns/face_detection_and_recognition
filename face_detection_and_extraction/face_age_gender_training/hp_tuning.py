"""
This script uses ray tune for hyper parameter optimization.
Most of the code is copied from the below link.
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
"""
import os
import argparse
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from utils import read_json
from model.model import ResMLP

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(config: dict):

    net = ResMLP(dropout=config['dropout'],
                 num_residuals_per_block=config['num_residuals_per_block'],
                 num_blocks=config['num_blocks'],
                 num_classes=config['num_classes'],
                 num_initial_features=config['num_initial_features'],
                 reduce_in_features=config["reduce_in_features"],
                 add_residual=config['add_residual'],
                 add_IC=config['add_IC'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    if config['criterion'] == 'mse':
        criterion = nn.MSELoss()
    elif config['criterion'] == 'cse':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError

    optimizer = optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config['gamma'])

    if config['gender_or_age_or_custom_video'].lower() == 'age':
        from data_loader.data_loaders import AgeDataLoader as DataLoader
    elif config['gender_or_age_or_custom_video'].lower() == 'gender':
        from data_loader.data_loaders import GenderDataLoader as DataLoader
    elif config['gender_or_age_or_custom_video'].lower() == 'custom_video':
        from data_loader.data_loaders import VideoFeatDataLoader as DataLoader
    else:
        raise ValueError

    trainloader = DataLoader(
        data_dir=config['data_dir'], batch_size=config['batch_size'], shuffle=True,
        validation_split=config['validation_split'], num_workers=config['cpus'], dataset=config['dataset'],
        num_classes=config['num_classes'], test_cross_val=None, training=None,
        limit_data=config['limit_data'])

    valloader = trainloader.split_validation()

    for epoch in range(config['max_num_epochs']):
        net.train()
        running_loss = 0.0
        epoch_steps = 0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if config['amp']:
                with autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if batch_idx % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        net.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        for val_batch_idx, (inputs, labels) in enumerate(valloader):
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                if config['amp']:
                    with autocast():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

        lr_scheduler.step()

    print("Finished Training")


def main(config_path: str):
    config = read_json(config_path)
    config['dropout'] = tune.choice(config['dropout'])
    config['num_residuals_per_block'] = tune.choice(
        config['num_residuals_per_block'])
    config['num_blocks'] = tune.choice(config['num_blocks'])
    config['batch_size'] = tune.choice(config['batch_size'])
    config['lr'] = tune.loguniform(*config['lr'])
    config['weight_decay'] = tune.loguniform(*config['weight_decay'])
    config['gamma'] = tune.loguniform(*config['gamma'])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config['max_num_epochs'],
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train),
        resources_per_trial={
            "cpu": config['cpus'], "gpu": config['gpus_per_trial']},
        config=config,
        num_samples=config['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='hp-tuning')
    args.add_argument('-c', '--config', default="config/hp_tuning_age.json",
                      help="Path to tuning json file (default: %(default)s)")
    config_path = args.parse_args().config
    main(config_path)
