from time import time
import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool, dispatch_clip_grad
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime
import shutil
import logging
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=3)

    # network related
    parser.add_argument('--backbone', type=str, default='cbam')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--method', type=str, default='P3ID')
    parser.add_argument('--data_dir', type=str, default="./dataset")
    parser.add_argument('--src_domain', type=str, default="A_train")
    parser.add_argument('--tgt_domain', type=str, default="B_train")
    parser.add_argument('--tgt_domain_valid', type=str, default="B_valid")
    parser.add_argument('--tgt_domain_test', type=str, default="B_test")
    parser.add_argument('--log_dir', type=str, default="./log/P3ID_{}".format(str(time.time())))
    parser.add_argument('--save_folder', type=str)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=True, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.03)
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.006)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--domain_loss_weight', type=float, default=0.1)
    parser.add_argument('--transfer_loss', type=str, default='lmmd')
    return parser

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, ref_domain, tgt_domain(valid), tgt_domain(test) data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_tgt_valid = os.path.join(args.data_dir, args.tgt_domain_valid)
    folder_tgt_test = os.path.join(args.data_dir, args.tgt_domain_test)

    mean_src, std_src = [0.1409, 0.5696, 0.5379], [0.0647, 0.0769, 0.0416]
    mean_tgt, std_tgt = [0.1430, 0.5715, 0.5363], [0.0674, 0.0807, 0.0436]
    mean_tgt_test, std_tgt_test = [0.1430, 0.5715, 0.5363], [0.0674, 0.0807, 0.0436]

    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, mean = mean_src, std = std_src, args = args, num_workers=args.num_workers,)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, mean = mean_tgt, std = std_tgt, args = args, num_workers=args.num_workers,)
    target_valid_loader, _ = data_loader.load_data(
        folder_tgt_valid, args.batch_size, infinite_data_loader=False, train=False, mean = mean_tgt_test, std = std_tgt_test, args = args,num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt_test, args.batch_size, infinite_data_loader=False, train=False, mean = mean_tgt_test, std = std_tgt_test, args = args,num_workers=args.num_workers)

    return source_loader, target_train_loader, target_valid_loader, target_test_loader, n_class


def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck,
        batch_size=args.batch_size).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.module.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.module.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            y_pred.extend(pred.data.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg


def train(source_loader, target_train_loader, target_valid_loader, target_test_loader, model, optimizer, lr_scheduler, args, writer):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)
    
    best_acc_valid = 0
    best_acc_valid_test = 0
    best_acc_test = 0
    stop = 0
    for e in range(1, args.n_epoch + 1):
        start_time = time.time()
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_domain = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.module.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for i in range(n_batch):
            data_source, label_source = next(iter_source)  # .next()
            data_target, _ = next(iter_target)  # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)

            clf_loss, transfer_loss, domain_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss + args.domain_loss_weight * domain_loss

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad:
                dispatch_clip_grad(model.module.parameters(), args.clip_grad, args.clip_mode, norm_type=2.0)
            optimizer.step()


            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_domain.update(domain_loss.item())
            train_loss_total.update(loss.item())
        end_time = time.time()
        epoch_time = end_time - start_time
        if args.local_rank == 0:
            print(epoch_time)

        if lr_scheduler:
            lr_scheduler.step()
        if args.local_rank == 0:
            writer.add_scalars('train/loss', {'clf_loss': clf_loss, 'transfer_loss': transfer_loss, 'domain_loss': domain_loss, 'loss': loss},
                           e)
            writer.close()

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, domain_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_domain.avg, train_loss_total.avg)
        # Test
        stop += 1
        acc_valid, loss_valid = test(model, target_valid_loader, args)
        acc_test, loss_test = test(model, target_test_loader, args)
        info += ', valid_loss {:4f}, valid_acc: {:.4f}, test_loss {:4f}, test_acc: {:.4f}'.format(loss_valid, acc_valid, loss_test, acc_test)

        if args.local_rank == 0:
            if best_acc_test <= acc_test:
                best_acc_test = acc_test
                stop = 0
                torch.save(model, 'checkpoint/'+args.method+'/'+args.save_folder+'/'+args.log_dir[6:]+'_'+args.src_domain + args.tgt_domain + args.tgt_domain_test+'/'+args.src_domain + args.tgt_domain + args.tgt_domain_test + '_best.pt')


            if best_acc_valid <= acc_valid:
                best_acc_valid = acc_valid
                best_acc_valid_test = acc_test
                stop = 0
                torch.save(model, 'checkpoint/'+args.method+'/'+args.save_folder+'/'+args.log_dir[6:]+'_'+args.src_domain + args.tgt_domain + args.tgt_domain_test+'/'+args.src_domain + args.tgt_domain + args.tgt_domain_test + '_best_valid.pt')


        if args.local_rank == 0:
            writer.add_scalars('test/acc', {'acc_target': acc_test, 'acc_valid': acc_valid}, e)
            writer.add_scalars('test/loss', {'loss_target': loss_test, 'loss_valid': loss_valid}, e)


        print(info)

    if args.local_rank == 0:
        print('Best global acc : {:.4f}, Best valid test acc : {:.4f}'.format(best_acc_test, best_acc_valid_test))


def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    parser = get_parser()
    args = parser.parse_args()

    # --------------log & results & backup--------------
    if args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

        # results
        result_dir = os.path.join('results', args.method, args.save_folder, args.log_dir[6:]+'_'+args.src_domain + args.tgt_domain + args.tgt_domain_test) 
        model_dir = os.path.join('checkpoint', args.method, args.save_folder, args.log_dir[6:] +'_'+args.src_domain + args.tgt_domain + args.tgt_domain_test)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        sys.stdout = Logger(os.path.join(result_dir,args.src_domain + args.tgt_domain + args.tgt_domain_test + '.txt'), sys.stdout) 


        print("Current Time:", datetime.now(),'\n')

    else:
        writer = None


    # --------------Check GPU & args--------------
    if args.local_rank == 0:
        if torch.cuda.is_available():
            logging.warning("Cuda is available!")
            if torch.cuda.device_count() > 1:
                logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
            else:
                logging.warning("Too few GPU!")
                return
        else:
            logging.warning("Cuda is not available! Exit!")
            return
    torch.cuda.set_device(args.local_rank)
    setattr(args, "device", torch.device('cuda',args.local_rank) if torch.cuda.is_available() else 'cpu')

    if args.local_rank == 0:
        print(args)

    torch.distributed.init_process_group(backend='nccl')

    set_random_seed(args.seed)


    # --------------dataset & model & train--------------
    source_loader, target_train_loader, target_valid_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_valid_loader, target_test_loader, model, optimizer, scheduler, args, writer)


if __name__ == "__main__":
    main()
