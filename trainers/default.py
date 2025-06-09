import time

import math
import torch
import tqdm
import copy

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

import wandb
import matplotlib.pyplot as plt
__all__ = ["train", "validate", "modifier"]



def train(train_loader, model, criterion, optimizer, epoch, args, reg, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )
    # switch to train mode
    model.train()


    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    j = len(train_loader)
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # 计算当前 step
        current_step = epoch * num_batches + i

        # 将分数用作正则化后进行微调
        # print('epoch',epoch)
        if args.finetune_aftertrain:
            model = multiply_scores_weights(args, epoch, model, current_step, reg, i, j)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

        # 将当前训练步的信息记录到 wandb
        wandb.log({'iter_train_acc1': acc1, 'iter_train_acc5': acc5, 'iter_train_loss': loss.item()}, step=current_step)

    return top1.avg, top5.avg



def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )
    # cp_model = copy.deepcopy(model)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # if args.finetune_aftertrain:
            #     cp_model = apply_mask_to_model(cp_model,args.prune_rate)

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))



        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    # # 手动释放模型副本
    # del cp_model  # 删除模型副本的变量
    # torch.cuda.empty_cache()  # 释放未使用的显存

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    pass




# import matplotlib.pyplot as plt


# def log_heatmap_combined(name, data, log_dict):
#     # 创建热力图
#     plt.figure(figsize=(8, 6))
#     plt.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
#     plt.colorbar()
#     plt.title(f"Heatmap of {name}")
#
#     # 把热力图保存为一个图片对象并添加到 log_dict
#     log_dict[f"{name}_reg_heatmap"] = wandb.Image(plt)
#     plt.close()  # 关闭图表以释放内存

def log_histogram_combined(name, data, log_dict):
    # 直接将张量数据添加为直方图到 log_dict 中
    log_dict[f"{name}_reg_histogram"] = wandb.Histogram(data)

def multiply_scores_weights(args, epoch, model, current_step, reg, i,j):
    e = math.exp(1)  # 自然底数
    log_dict = {}  # 用来存储所有模块的记录

    for name, m in model.named_modules():
        if hasattr(m, 'scores') and m.weight is not None and m.scores is not None:
            scores_temp = m.scores.detach().clone()
            # 绝对值和归一化步骤已经在 switch_to_wt() 方法中完成 弃用下面的步骤
                # # 先取绝对值 再归一化
                # # 不然负的不就被归零了
                # # 取绝对值
                # scores_temp = abs(scores_temp)
                #
                # # 将 scores_temp 的值归一化到 (0, 1) 范围
                # scores_temp = (scores_temp - scores_temp.min()) / (scores_temp.max() - scores_temp.min())
                #
                # # 为了确保值在 (0, 1) 之间，而不是 0 和 1，稍微缩放一下
                # scores_temp = scores_temp * 0.99 + 0.01

            # 计算正则化增长权重
            # weight_start = (
            #         1 / (e - 1)
            #         * args.scores_lambda
            #         * (math.exp((i + 1) ** args.exp_custom_exponents / j ** args.exp_custom_exponents) - 1)
            # )
            weight_start = (
                    1 / (e - 1)
                    * args.scores_lambda
                    * (math.exp((epoch + i / j + 1) ** args.exp_custom_exponents / args.epochs ** args.exp_custom_exponents) - 1)
            )
            # weight_start = (
            #         1 / (math.e - 1)
            #         * 0.1
            #         * (math.exp((epoch + 1) ** args.exp_custom_exponents / args.epochs ** args.exp_custom_exponents) - 1)
            # )

            # reg[name] += weight_start * (1 - scores_temp)
            # test bench mark RST
            reg[name] += weight_start * (1 - scores_temp)

            reg[name] = torch.clamp(reg[name], max=1)

            # 将卷积核展平为行的形式，列是 Kernel Height, Kernel Width, Input Channel
            # reshaped_reg = reg[name].view(reg[name].shape[0], -1).cpu().numpy()

            # 添加热力图到 log_dict
            # log_heatmap_combined(name, reshaped_reg, log_dict)

            # 添加直方图到 log_dict
            log_histogram_combined(name, reg[name].cpu().numpy(), log_dict)

            # scores_weight_grad = m.weight * reg[name]
            scores_weight_grad = m.weight * reg[name]
            m.weight.grad += scores_weight_grad

    # 在每个训练 step 内只记录一次所有模块的数据
    wandb.log(log_dict, step=current_step)

    return model


