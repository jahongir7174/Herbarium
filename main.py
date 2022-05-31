import argparse
import copy
import csv
import os
import random
import warnings

import numpy
import torch
import tqdm
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = os.path.join('/Projects', 'Dataset', 'Herbarium')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def set_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def learning_rate(args):
    lr = 0.256 / 1024
    if not args.distributed:
        return args.batch_size * lr
    else:
        return args.batch_size * lr * args.world_size


def batch(images, target, model, criterion):
    images = images.cuda()
    target = target.cuda()

    with torch.cuda.amp.autocast():
        output = model(images)
    return criterion(output, target)


def cut_mix(images, target, model, criterion):
    shape = images.size()
    index = torch.randperm(shape[0]).cuda()
    alpha = numpy.sqrt(1. - numpy.random.beta(1.0, 1.0))

    w = numpy.int(shape[2] * alpha)
    h = numpy.int(shape[3] * alpha)

    # uniform
    c_x = numpy.random.randint(shape[2])
    c_y = numpy.random.randint(shape[3])

    x1 = numpy.clip(c_x - w // 2, 0, shape[2])
    y1 = numpy.clip(c_y - h // 2, 0, shape[3])
    x2 = numpy.clip(c_x + w // 2, 0, shape[2])
    y2 = numpy.clip(c_y + h // 2, 0, shape[3])

    images[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]

    alpha = 1 - ((x2 - x1) * (y2 - y1) / (shape[-1] * shape[-2]))

    images = images.cuda()
    target = target.cuda()

    with torch.cuda.amp.autocast():
        output = model(images)
    return criterion(output, target) * alpha + criterion(output, target[index]) * (1. - alpha)


def train(args):
    model = nn.create()
    ema_m = nn.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = nn.RMSprop(util.weight_decay(model), learning_rate(args))

    if not args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])

    scheduler = nn.StepLR(optimizer)
    criterion = nn.CrossEntropyLoss().cuda()

    with open(f'./weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
            writer.writeheader()

        sampler = None
        dataset = Dataset(os.path.join(data_dir, 'train'),
                          transforms.Compose([util.Resize(args.input_size),
                                              util.RandomAugment(mean=9, n=1),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), normalize,
                                              util.Cutout()]))

        if args.distributed:
            sampler = data.distributed.DistributedSampler(dataset)

        loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                                 sampler=sampler, num_workers=8, pin_memory=True)

        model.train()

        for epoch in range(args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            p_bar = loader
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(loader, total=len(loader))

            m_loss = util.AverageMeter()

            for images, target in p_bar:
                loss = cut_mix(images, target, model, criterion)

                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema_m.update(model)
                torch.cuda.synchronize()

                if args.distributed:
                    loss = loss.data.clone()
                    torch.distributed.all_reduce(loss)
                    loss /= args.world_size

                loss = loss.item()
                m_loss.update(loss, images.size(0))

                if args.local_rank == 0:
                    desc = ('%10s' + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), loss)
                    p_bar.set_description(desc)

            scheduler.step(epoch + 1)

            if args.local_rank == 0:
                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'loss': str(f'{m_loss.avg:.3f}')})

                state = {'model': copy.deepcopy(ema_m.model).half()}
                torch.save(state, './weights/model.pt')

                del state

        del loader
        del sampler
        del dataset

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(args):
    from utils.dataset import TestDataset

    model = torch.load('weights/model1.pt', 'cuda')['model'].float()
    model.half()
    model.eval()

    dataset = TestDataset(data_dir,
                          transforms.Compose([transforms.Resize(size=args.input_size),
                                              transforms.CenterCrop(args.input_size),
                                              transforms.ToTensor(), normalize]))
    loader = data.DataLoader(dataset, 32, num_workers=4)
    with open(f'submission.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Predicted'])
        writer.writeheader()
        for images, indices in tqdm.tqdm(loader):
            images = images.cuda()
            images = images.half()

            with torch.no_grad():
                for index, output in zip(indices, model(images)):
                    output = int(output.argmax().cpu().numpy())
                    output = dataset.idx_to_cls[output]

                    writer.writerow({'Id': int(index), 'Predicted': output})


def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.getenv('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
