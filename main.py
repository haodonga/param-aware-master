import os
import pathlib
import random
import time
import tqdm
import re

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    freeze_model_subnet,
    unfreeze_model_subnet,
    unfreeze_model_weights, save_finetune_checkpoint

)
from utils.schedulers import get_policy


from args import args
import importlib

import data
import models

import copy
import wandb

def main():
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    reg=None
    args.gpu = None
    train, validate, modifier = get_trainer(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        print('resume:',args.resume)
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    # progress_overall = ProgressMeter(
    #     1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    # )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    if args.finetune:
        is_finetune = "open_finetune"
    else:
        is_finetune = "close_finetune"

    scores_lambda = str(args.scores_lambda)
    batch_sizes = str(args.batch_size)
    epochs = str(args.epochs)
    prune_rates = str(args.prune_rate)
    #初始化wandb
    wandb_name = '_'.join([is_finetune, scores_lambda, "unif", prune_rates, args.set, args.arch, batch_sizes, epochs, "upperbond"])

    wandb.init(project='EP改造计划',name=wandb_name,config=vars(args))

    wandb.watch(model, log='all', log_freq=1, log_graph=True)

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, reg, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        wandb.log({'epoch': epoch, 'epoch_train_acc1': train_acc1, 'epoch_train_acc5': train_acc5, 'epoch_test_acc1': acc1,
                   'epoch_test_acc5': acc5})

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        # 为了减少训练时间 暂时关闭模型保存
        # save = ((epoch % args.save_every) == 0) and args.save_every > 0
        # if is_best or save or epoch == args.epochs - 1:
        #     if is_best:
        #         print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
        #
        #     save_checkpoint(
        #         {
        #             "epoch": epoch + 1,
        #             "arch": args.arch,
        #             "state_dict": model.state_dict(),
        #             "best_acc1": best_acc1,
        #             "best_acc5": best_acc5,
        #             "best_train_acc1": best_train_acc1,
        #             "best_train_acc5": best_train_acc5,
        #             "optimizer": optimizer.state_dict(),
        #             "curr_acc1": acc1,
        #             "curr_acc5": acc5,
        #         },
        #         is_best,
        #         filename=ckpt_base_dir / f"epoch_{epoch}.state",
        #         save=save,
        #     )

        epoch_time.update((time.time() - end_epoch) / 60)
        # progress_overall.display(epoch)
        # progress_overall.write_to_tensorboard(
        #     writer, prefix="diagnostics", global_step=epoch
        # )

        if args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", args.prune_rate, epoch)

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    if args.finetune:
        print("开始用分数正则化微调")
        args.finetune_aftertrain = True
        mask = finetune(model, args, data, criterion, writer)
        after_finetune_test(data.val_loader, model, mask, args)

    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")
        # 对于分数的剪枝在getmodel时就开始了，这里我将它注释 实验观察score
        set_model_prune_rate(model, prune_rate=0.1)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


def finetune(model, args, data, criterion, writer):
    reg = {}
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    #解冻权重，冻结分数
    # model = freeze_model_subnet(model)
    # model = unfreeze_model_weights(model)
    model, mask = switch_to_wt(model)

    optimizer = get_optimizer(args, model)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
    train, validate, modifier = get_trainer(args)
    args.finetune = True
    if args.finetune_aftertrain:
        for name, m in model.named_modules():
            if hasattr(m, 'scores'):
                reg[name] = torch.zeros_like(m.scores)
    for epoch in range(args.epochs, args.epochs*2):
        lr_policy(epoch - args.epochs, iteration=None)
        # cur_lr = get_lr(optimizer)

        # train for one epoch
        # start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args,reg ,writer=writer
        )
        # train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        # start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        # 测试带mask的acc
        model_copy = copy.deepcopy(model)
        after_finetune_test(data.val_loader, model_copy, mask, args)
        # validation_time.update((time.time() - start_validation) / 60)
        wandb.log(
            {'epoch': epoch, 'epoch_train_acc1': train_acc1, 'epoch_train_acc5': train_acc5, 'epoch_test_acc1': acc1,
             'epoch_test_acc5': acc5})
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {args.ckpt_base_dir / 'afterfinetune_model_best.pth'}")

            save_finetune_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=args.ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        # epoch_time.update((time.time() - end_epoch) / 60)
        # progress_overall.display(epoch)
        # progress_overall.write_to_tensorboard(
        #     writer, prefix="diagnostics", global_step=epoch
        # # )

        # if args.conv_type == "SampleSubnetConv":
        #     count = 0
        #     sum_pr = 0.0
        #     for n, m in model.named_modules():
        #         if isinstance(m, SampleSubnetConv):
        #             # avg pr across 10 samples
        #             pr = 0.0
        #             for _ in range(10):
        #                 pr += (
        #                     (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
        #                     .float()
        #                     .mean()
        #                     .item()
        #                 )
        #             pr /= 10.0
        #             writer.add_scalar("pr/{}".format(n), pr, epoch)
        #             sum_pr += pr
        #             count += 1
        #
        #     args.prune_rate = sum_pr / count
        #     writer.add_scalar("pr/average", args.prune_rate, epoch)

        # writer.add_scalar("test/lr", cur_lr, epoch)
        # end_epoch = time.time()
    return mask


#原始的测试代码
# def after_finetune_test(val_loader, model, mask, args):
#     print("微调后测试Acc")
#     # batch_time = AverageMeter("Time", ":6.3f", write_val=False)
#     losses = AverageMeter("Loss", ":.3f", write_val=False)
#     top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
#     top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
#     # progress = ProgressMeter(
#     # len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
#     # )
#     model.eval()
#     # 打印稀疏度
#     # print_nonzeros(model)
#     # model = apply_mask_to_model(model, args.prune_rate)
#     model = prune_by_percentile(model, mask, args.prune_rate)
#
#     with torch.no_grad():
#         # end = time.time()
#         for i, (images, target) in tqdm.tqdm(
#                 enumerate(val_loader), ascii=True, total=len(val_loader)
#         ):
#             # 检查数据加载情况
#             # print(f"Batch {i}: images.shape = {images.shape}, target.shape = {target.shape}")
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu, non_blocking=True)
#
#             target = target.cuda(args.gpu, non_blocking=True)
#
#             output = model(images)
#
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#
#             wandb.log({'LAST_acc1': acc1, 'LAST_acc5': acc5})
#
#         print(f"微调之后最终的Top-1 Accuracy是: {acc1.item():.2f}%")
#         print(f"微调之后最终的Top-5 Accuracy: {acc5.item():.2f}%")
#
#gpt的测试代码
def after_finetune_test(val_loader, model, mask, args):
    print("微调后测试Acc")

    # 创建计量器
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)

    # 设置模型为评估模式
    model.eval()

    # 对模型进行剪枝
    model = prune_by_percentile(model, mask, args.prune_rate)

    # 用于累积准确率
    total_acc1 = 0
    total_acc5 = 0

    # 不计算梯度
    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader)):
            # 将数据移到GPU（如果有的话）
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # 通过模型获取输出
            output = model(images)

            # 计算Top-1和Top-5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 累积准确率
            total_acc1 += acc1[0]  # 取出acc1的值
            total_acc5 += acc5[0]  # 取出acc5的值

        # 计算平均准确率
        avg_acc1 = total_acc1 / len(val_loader)
        avg_acc5 = total_acc5 / len(val_loader)

        # 打印最终准确率
        print(f"微调之后最终的Top-1 Accuracy是: {avg_acc1:.2f}%")
        print(f"微调之后最终的Top-5 Accuracy是: {avg_acc5:.2f}%")

        # 记录到wandb
        wandb.log({'LAST_acc1': avg_acc1, 'LAST_acc5': avg_acc5})
#姚晨曦的测试代码
# def after_finetune_test(val_loader, model, mask, args):
#
#     print("微调后测试Acc")
#     # batch_time = AverageMeter("Time", ":6.3f", write_val=False)
#     losses = AverageMeter("Loss", ":.3f", write_val=False)
#     top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
#     top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
#     # progress = ProgressMeter(
#     #     len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
#     # )
#     model.eval()
#     #将mask应用的model中
#     model = prune_by_percentile(model, mask, args.prune_rate)
#     #打印稀疏度
#     # print_nonzeros(model)
#     with torch.no_grad():
#         # end = time.time()
#         for i, (images, target) in tqdm.tqdm(
#                 enumerate(val_loader), ascii=True, total=len(val_loader)
#         ):
#
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu, non_blocking=True)
#
#             target = target.cuda(args.gpu, non_blocking=True)
#
#             output = model(images)
#             # loss = criterion(output, target)
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             # losses.update(loss.item(), images.size(0))
#             top1.update(acc1.item(), images.size(0))
#             top5.update(acc5.item(), images.size(0))
#
#             wandb.log({'after_finetune_test_acc1': acc1, 'after_finetune_test_acc5': acc5})
#
#     print(f"微调之后最终的Top-1 Accuracy是: {top1.avg}")
#     print(f"微调之后最终的Top-5 Accuracy: {top5.avg}")
#
#     wandb.log({"Last_Acc1":top1.avg,"Last_Acc5":top5.avg})

#根据分数和prune_rate生成掩码
def generate_mask(scores, prune_rate):
    # 测试：分数取绝对值
    scores = abs(scores)
    # 1. 展平 scores，并进行排序
    flat_scores = scores.flatten()
    sorted_scores, idx = flat_scores.sort()

    # 2. 计算需要保留的参数数量
    num_to_retain = int((1 - prune_rate) * flat_scores.numel())

    # 3. 创建一个全 0 掩码
    mask = torch.zeros_like(flat_scores)

    # 4. 将前 num_to_retain 个高分部分置为 1
    mask[idx[-num_to_retain:]] = 1

    # 5. 重新调整 mask 形状，恢复到与 scores 相同的形状
    return mask.view_as(scores)

#将mask应用到model中
def prune_by_percentile(model: nn.Module, mask, percent: float):


    # # 生成掩码
    # for name, param in model.named_parameters():
    #     if name.endswith('scores'):
    #
    #         mask[name] = torch.ones_like(param.data)  # 为每个 score 参数生成掩码

    # 应用剪枝
    # for name, param in model.named_parameters():
    #     if name.endswith('scores'):
    #         tensor = param.data
    #         alive = tensor[tensor != 0]  # 获取非零权重
    #
    #         # 计算百分位数值
    #         percentile_value = torch.quantile(alive.abs(), percent)
    #
    #         # 生成新的掩码
    #         new_mask = (tensor.abs() >= percentile_value).float() * mask[name].to(tensor.device)
    #
    #         # 应用新的掩码并更新权重
    #         param.data.mul_(new_mask)
    #
    #         mask[name] = new_mask.clone()  # 更新掩码

    # 将掩码应用到对应的权重上
    for name, param in model.named_parameters():
        if name.endswith('weight'):
            score_name = name.replace('weight', 'scores')
            if score_name in mask:
                print(f'Applying mask to {name}')
                param.data.mul_(mask[score_name].to(param.device))

    return model


def apply_mask_to_model(model,prune_rate):
    # 遍历模型所有子模块（子层）
    for name, module in model.named_modules():
        # 检查是否包含 `scores` 张量
        if hasattr(module, "scores") and module.weight is not None:
            with torch.no_grad():
                # 生成当前层的剪枝掩码
                mask = generate_mask(module.scores.detach().clone(), prune_rate)
                # 应用掩码到权重上
                module.weight.data = module.weight.data * mask

    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 优化了代码逻辑 提高执行效率
def switch_to_wt(model):
    print('Switching to weight training by switching off requires_grad for scores and switching it on for weights.')
    # ========================================全局剪枝===============================================

    # Step 1: Collect all `scores` data across the model
    # all_scores = []
    # for name, params in model.named_parameters():
    #     if "scores" in name:
    #         # 取绝对值并归一化
    #         params.data = abs(params.data)  # 取绝对值
    #         params_min = params.data.min()
    #         params_max = params.data.max()
    #
    #         # 将 scores 参数的值归一化到 (0, 1) 范围
    #         params.data = (params.data - params_min) / (params_max - params_min)
    #
    #         # 缩放到 (0.01, 0.99) 范围，避免0和1的极端值
    #         params.data = params.data * 0.99 + 0.01
    #
    #         # 将归一化后的 scores 添加到 all_scores 列表中
    #         all_scores.append(params.data)  # 添加归一化的 scores

    # Concatenate all scores to calculate the global threshold
    # all_scores = torch.cat([score.view(-1) for score in all_scores])  # Flatten and concatenate all scores
    # threshold = torch.quantile(all_scores, args.prune_rate)  # Calculate the threshold for pruning
    # ========================================全局剪枝===============================================

    # Initialize a mask dictionary to store the generated masks
    mask = {}

    # Iterate over the model parameters
    for name, params in model.named_parameters():
        if name.endswith('.weight'):
            # Enable gradient for weight parameters
            params.requires_grad = True
        elif "scores" in name:
            #========================================局部剪枝===============================================
            all_scores = []
            # 取绝对值并归一化
            params.data = abs(params.data)  # 取绝对值
            params_min = params.data.min()
            params_max = params.data.max()

            # 将 scores 参数的值归一化到 (0, 1) 范围
            params.data = (params.data - params_min) / (params_max - params_min)

            # 缩放到 (0.01, 0.99) 范围，避免0和1的极端值
            params.data = params.data * 0.99 + 0.01

            # 将归一化后的 scores 添加到 all_scores 列表中
            all_scores.append(params.data)  # 添加归一化的 scores
            # Concatenate all scores to calculate the global threshold
            all_scores = torch.cat([score.view(-1) for score in all_scores])  # Flatten and concatenate all scores
            threshold = torch.quantile(all_scores, args.prune_rate)  # Calculate the threshold for pruning
            #== == == == == == == == == == == == == == == == == == == == 局部剪枝 == == == == == == == == == == == == == == == == == == == == == == == =

            # Disable gradient for score parameters and create a mask
            # print('取scores绝对值在前10%分位数的scores置为1')
            params.requires_grad = False
            scores = params.data.abs()  # Get absolute values of scores
            mask[name] = (scores >= threshold).float()  # Create mask and apply threshold in one step
            params.data[scores >= threshold] = 1  # Set values above threshold to 1
            # test bench mark RST
            params.data[scores < threshold] = 0  # Set values above threshold to 0

        else:
            # Disable gradient for other parameters (e.g., biases)
            params.requires_grad = False

    return model, mask

if __name__ == "__main__":
    main()
