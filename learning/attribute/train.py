import sys
import os
import argparse
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from datasets.market import Market1501Dataset
from nets import EDNet

EPOCHS = 15
VALIDATION_BATCH = 5
TRAIN_BATCH = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Training script for attribute model")
    parser.add_argument(
        "--logdir",
        default='/tmp/out_logs',
        type=str,
        help="Path to logging directory for TensorBoard",
    )
    parser.add_argument(
        "--root",
        default="/tmp",
        type=str,
        help="Path to logging directory for TensorBoard",
    )
    parser.add_argument(
        "--epochs",
        default=EPOCHS,
        type=int,
        help="Path to logging directory for TensorBoard",
    )
    return parser.parse_args(argv)


def _modes():
    return ['train', 'test']


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


def train(data_root, epochs, log_dir):
    input_transforms = torchvision.transforms.Compose([
        lambda x: x / 255.0,
        torchvision.transforms.ToTensor(), lambda x: x.type(torch.FloatTensor)
    ])
    # Skip the first 2 elements in the lable which are the ID, age and subtract 1
    # to make it one-hot encoded
    target_transforms = torchvision.transforms.Compose([lambda x: x[2:] - 1])
    loaders = {}
    for mode in _modes():
        dataset = Market1501Dataset(root=data_root,
                                    train=mode == 'train',
                                    input_transforms=input_transforms,
                                    target_transforms=target_transforms)
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=_batch(mode),
                                                    drop_last=True)
    net = EDNet(input_shape=(3, 128, 64), num_classes=26, num_downsamples=3)
    print(net)
    net = net.to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    global_step = 0
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir,
                             datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
        flush_secs=20,
        filename_suffix=datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    for epoch in range(epochs):
        running_loss = {'train': 0.0, 'test': 0.0}
        for mode in _modes():
            print("Running {} on {} samples".format(mode, len(loaders[mode])))
            if mode == "test":
                net.eval()
            else:
                net.train()
            for i, data in enumerate(loaders[mode]):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss[mode] += loss.item()
                if mode == "train":
                    loss.backward()
                    optimizer.step()
                if mode == "train" and i % 500 == 0:
                    writer.add_scalar('loss/500th_iter_train_loss',
                                      loss.item(), global_step)
                    print("Training loss, iter {}: {}".format(
                        i, running_loss['train'] / (i + 1)))
                if mode == 'train':
                    global_step += 1
            writer.add_scalars('loss/epoch_loss', {
                '{}_loss'.format(mode):
                running_loss[mode] / len(loaders[mode])
            }, global_step)
        print("Epoch {}: Train Loss {}, Validation Loss {}.".format(
            epoch, running_loss['train'] / len(loaders['train']),
            running_loss['test'] / len(loaders['test'])))
    writer.close()


def main(argv):
    args = parse_args(argv)
    train(args.root, args.epochs, args.logdir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
