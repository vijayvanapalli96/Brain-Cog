import braincog
from braincog.base.utils.criterions import UnilateralMse
from braincog.datasets.datasets import *
from braincog.model_zoo.vgg_snn import VGG_SNN
from braincog.base.node import *
from braincog.utils import accuracy
import argparse
import json
import torch
import os


def get_config():
    parser = argparse.ArgumentParser(description='SNN Training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='training batch size')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate')
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of data loading workers')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='path to dataset')
    parser.add_argument(
        '--model',
        type=str,
        default='vgg16',
        help='model name')
    parser.add_argument(
        '--step',
        type=int,
        default=8,
        help='snn step')
    parser.add_argument(
        '--device',
        default='4',
        type=str,
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--node_type',
        type=str,
        default='LIFNode',
        help='node type in network')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='resume training from a checkpoint')
    parser.add_argument(
        "--dataset_params",
        type=json.loads,
        default={},
        help='A dict according to the official train_classifier API')
    parser.add_argument(
        "--model_params",
        type=json.loads,
        default={},
        help='A dict according to the official train_classifier API')


    args = parser.parse_args()
    print(args)
    return args


def validate(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy


if __name__ == '__main__':
    args = get_config()
    device = torch.device('cuda:%s' %
                          args.device) if torch.cuda.is_available() else 'cpu'
    # ---- get dataset ----
    if args.dataset == 'cifar10':
        train_loader, test_loader, _, _ = get_cifar10_data(
            args.batch_size, num_workers=args.workers)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader, _, _ = get_cifar100_data(
            args.batch_size, num_workers=args.workers)
        num_classes = 100
    elif args.dataset == 'mnist':
        train_loader, test_loader, _, _ = get_mnist_data(
            args.batch_size, num_workers=args.workers)
        num_classes = 10
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader, _, _ = get_fashion_data(
            args.batch_size, num_workers=args.workers)
        num_classes = 10
    elif args.dataset == 'dvsg':
        train_loader, test_loader, _, _ = get_dvsg_data(
            args.batch_size, step=args.step)
        num_classes = 11
    elif args.dataset == 'dvsc10':
        train_loader, test_loader, _, _ = get_dvsc10_data(
            args.batch_size, step=args.step)
        num_classes = 10
    elif args.dataset == "my_custom_data":
        # Point the data loader to our newly grouped dataset
        args.dataset_params['data_path'] = 'data/my_grouped_recordings'
        train_loader, test_loader, _, _, num_classes = get_my_custom_data(
            args.batch_size,
            step=args.step,
            **args.dataset_params
        )
    else:
        raise('Not implemented.')

    # ---- get model ----
    node_type = getattr(braincog.base.node, args.node_type)
    model = VGG_SNN(
        num_classes=num_classes,
        step=args.step,
        node_type=node_type,
        dataset=args.dataset,
        **args.model_params
    )
    model = model.to(device)

    # ---- get criterion ----
    criterion = UnilateralMse()

    # ---- get optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- get lr scheduler ----
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ---- training ----
    for epoch in range(args.epochs):
        print("Epoch: %d" % epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        validate(model, test_loader, device, criterion)
        lr_scheduler.step()

    # ---- save model ----
    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(),
               os.path.join(save_path, f'{args.model}-{args.dataset}.pth'))