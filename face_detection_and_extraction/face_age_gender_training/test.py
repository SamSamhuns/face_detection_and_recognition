import argparse
from tqdm import tqdm

import torch
import numpy as np
from torch.cuda.amp import autocast  # for float16 mixed point precision

from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def test(config: ConfigParser, checkpoint: str) -> dict:
    """Save test results"""
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir=config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        num_workers=config['data_loader']['args']['num_workers'],
        dataset=config['data_loader']['args']['dataset_test'],
        num_classes=config['data_loader']['args']['num_classes'],
        training=False,
    )
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info(f'Loading checkpoint: {checkpoint} ...')
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    if config['n_gpu'] > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(data_loader)):
            data, target = data.to(device), target.to(device)

            if config['amp']:
                with autocast():
                    output = model(data)
                    loss = loss_fn(output, target)
            else:
                output = model(data)
                loss = loss_fn(output, target)

            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    if n_samples == 0:
        raise Exception(f"Test dataset {config['data_loader']['args']['dataset_test']} is missing or empty")

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(f"test: {log}")
    return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/train_age.json", type=str,
                      help='config file path (default: %(default)s)')
    args.add_argument('-r', '--resume', default=None, type=str, required=True,
                      help='path to checkpoint for testing (default: %(default)s)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: %(default)s)')

    config = ConfigParser.from_args(args)
    test(config, config.resume)
