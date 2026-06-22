import os
import torch
import torch.nn.functional as F
from data import *
from utils import *
from reconstruction_model import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Object_Mask import *
import glob
import sys
import warnings
import argparse
import random
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from contextlib import contextmanager

torch.autograd.set_detect_anomaly(False)

# 设置随机种子函数
def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def cpu_data_generator(seed):
    try:
        generator = torch.Generator(device='cpu')
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def loader_options(num_workers, pin_memory):
    options = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        options['persistent_workers'] = True
    return options


def create_grad_scaler(enabled):
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def pooled_channel_feature(feature):
    return feature.mean(dim=(0, 2, 3, 4))


def negative_kl_divergence(source, target):
    return -F.kl_div(
        F.log_softmax(source, dim=0),
        F.softmax(target, dim=0),
        reduction='sum'
    )


def smooth_video_scores(scores, sigma):
    scores = np.asarray(scores, dtype=np.float64)
    if sigma <= 0 or scores.size < 2:
        return scores.tolist()
    return gaussian_filter1d(scores, sigma=sigma, mode='nearest').tolist()


def reconstruction_score(output_frame, input_frame, mode, patch_size,
                         global_weight=0.0):
    residual = (output_frame - input_frame).pow(2).mean(dim=0, keepdim=True)
    global_mse = residual.mean()

    if mode == 'psnr':
        return psnr(global_mse.item()), global_mse.item()
    if mode == 'mse':
        return global_mse.item(), global_mse.item()
    if mode == 'max_patch':
        kernel = min(patch_size, residual.shape[-2], residual.shape[-1])
        patch_scores = F.avg_pool2d(
            residual.unsqueeze(0),
            kernel_size=kernel,
            stride=kernel
        )
        patch_score = patch_scores.max()
        score = (1.0 - global_weight) * patch_score + global_weight * global_mse
        return score.item(), global_mse.item()
    raise ValueError(f"Unsupported score mode: {mode}")


def cosine_decay_multiplier(epoch, decay_epochs, min_lr_ratio):
    progress = min(epoch, decay_epochs) / max(1, decay_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        null_device = open('nul' if os.name == 'nt' else '/dev/null', 'w')
        sys.stdout = null_device
        sys.stderr = null_device
        yield
    finally:
        null_device.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def efficient_train_step(imgs, models, optimizer, scaler, loss_func_mse, args,
                         device, channel_in, sigma, dependency_state, amp_enabled):
    En, Den, Dep, mem, mem_p = models
    net_in = imgs.to(device, non_blocking=True)
    pseudo_mask = torch.rand(net_in.size(0), device=device) < args.pseudo_anomaly
    pseudo_indices = pseudo_mask.nonzero(as_tuple=False).flatten()
    pseudo_type = None

    if pseudo_indices.numel() > 0:
        pseudo_type = random.choice(args.pseudo_types)
        pseudo_source = net_in.index_select(0, pseudo_indices).clone()

        if pseudo_type == 'gaussian':
            pseudo_source = gaussian(pseudo_source, 1, 0, sigma)
        elif pseudo_type == 'shuffle':
            pseudo_source = genMotionAnoSmps(pseudo_source)
        elif pseudo_type == 'jump':
            try:
                imgs_jump = next(dependency_state['jump_iter'])
            except StopIteration:
                dependency_state['jump_iter'] = iter(dependency_state['jump_loader'])
                imgs_jump = next(dependency_state['jump_iter'])
            imgs_jump = imgs_jump.to(device, non_blocking=True)
            pseudo_source = imgs_jump.index_select(0, pseudo_indices)
        elif pseudo_type == 'object':
            if dependency_state['yolo_model'] is None:
                with suppress_output():
                    yolo_model, cifar_loader, cifar_iter = init_dependencies(
                        cifar_path='./dataset/cifar100',
                        yolo_cfg='Yolov3/yolov3/cfg/yolov3-spp.cfg',
                        yolo_weights='Yolov3/yolov3/weights/yolov3-spp-ultralytics.pt',
                        device=device,
                        channel_in=channel_in
                    )
                dependency_state.update({
                    'yolo_model': yolo_model,
                    'cifar_loader': cifar_loader,
                    'cifar_iter': cifar_iter,
                })

            with suppress_output():
                with torch.no_grad():
                    pseudo_source, dependency_state['cifar_iter'] = generate_pseudo_anomalies(
                        net_in=pseudo_source,
                        yolo_model=dependency_state['yolo_model'],
                        cifar_loader=dependency_state['cifar_loader'],
                        cifar_iter=dependency_state['cifar_iter'],
                        device=device,
                        channel_in=channel_in
                    )

        net_in = net_in.clone()
        net_in.index_copy_(0, pseudo_indices, pseudo_source)

    normal_indices = (~pseudo_mask).nonzero(as_tuple=False).flatten()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast('cuda', enabled=amp_enabled):
        fea, x2, x1, x0 = En(net_in)
        out = torch.empty_like(net_in, dtype=fea.dtype)
        zero_loss = fea.sum() * 0.0
        sparsity_sum = zero_loss
        loss_feas = zero_loss
        kl_fea = zero_loss
        kl_out3 = zero_loss
        normal_features = None
        pseudo_features = None
        normal_decoder_feature = None
        pseudo_decoder_feature = None

        if normal_indices.numel() > 0:
            mem_out_n = mem(fea.index_select(0, normal_indices))
            out_n, _, _, out_n3 = Den(
                mem_out_n["x3_out"],
                x2.index_select(0, normal_indices),
                x1.index_select(0, normal_indices),
                x0.index_select(0, normal_indices)
            )
            out.index_copy_(0, normal_indices, out_n)
            att_weight_n = mem_out_n["att_weight"]
            entropy_n = torch.mean(torch.sum(
                -att_weight_n * torch.log(att_weight_n + 1e-12), dim=1
            ))
            sparsity_sum = sparsity_sum + entropy_n * normal_indices.numel()
            normal_features = pooled_channel_feature(mem_out_n["x3_out"])
            normal_decoder_feature = pooled_channel_feature(out_n3)

        if pseudo_indices.numel() > 0:
            mem_out_p = mem_p(fea.index_select(0, pseudo_indices))
            out_p, _, _, out_p3 = Dep(
                mem_out_p["x3_out"],
                x2.index_select(0, pseudo_indices),
                x1.index_select(0, pseudo_indices),
                x0.index_select(0, pseudo_indices)
            )
            out.index_copy_(0, pseudo_indices, out_p)
            att_weight_p = mem_out_p["att_weight"]
            entropy_p = torch.mean(torch.sum(
                -att_weight_p * torch.log(att_weight_p + 1e-12), dim=1
            ))
            sparsity_sum = sparsity_sum + entropy_p * pseudo_indices.numel()
            pseudo_features = pooled_channel_feature(mem_out_p["x3_out"])
            pseudo_decoder_feature = pooled_channel_feature(out_p3)

        loss_mse = loss_func_mse(out, net_in).flatten(1).mean(dim=1)
        signed_mse = torch.where(pseudo_mask, -loss_mse, loss_mse).mean()
        loss_sparsity = sparsity_sum / net_in.size(0)

        if normal_features is not None and pseudo_features is not None:
            loss_feas = -torch.mean(torch.abs(normal_features - pseudo_features))
            kl_fea = negative_kl_divergence(normal_features, pseudo_features)
            kl_out3 = negative_kl_divergence(
                normal_decoder_feature, pseudo_decoder_feature
            )

        diff_weight = 0.0 if args.disable_diff_loss else args.diff_loss_weight
        sparsity_weight = (
            0.0 if args.disable_sparsity_loss else args.loss_m_weight
        )
        loss = signed_mse \
               + loss_feas * diff_weight * args.lambda_l1 \
               + kl_fea * diff_weight * args.lambda_kl_fea \
               + kl_out3 * diff_weight * args.lambda_kl_dec \
               + loss_sparsity * sparsity_weight

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss, {
        'normal_samples': normal_indices.numel(),
        'pseudo_samples': pseudo_indices.numel(),
        'pseudo_type': pseudo_type,
    }


def train_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")
    if args.h % 8 != 0 or args.w % 8 != 0:
        raise ValueError("--h and --w must be divisible by 8 for the three encoder downsampling stages")
    if not 0.0 <= args.score_global_weight <= 1.0:
        raise ValueError("--score_global_weight must be between 0 and 1")
    channel_in = 1
    if args.dataset_type == 'ped2':
        channel_in = 1
        train_folder = os.path.join(args.data_root, 'UCSDped2', 'Train')
        test_folder = os.path.join(args.data_root, 'UCSDped2', 'Test')
        learning_rate = args.learning_rate_ped2
    if args.dataset_type == 'avenue':
        channel_in = 3
        train_folder = os.path.join(args.data_root, 'Avenue', 'Train')
        test_folder = os.path.join(args.data_root, 'Avenue', 'Test')
        learning_rate = args.learning_rate_avenue
    if args.dataset_type == 'shanghai':
        channel_in = 3
        learning_rate = 1e-4
        shanghai_root = os.path.join(args.data_root, 'shanghaitech')
        if not os.path.isdir(shanghai_root):
            shanghai_root = os.path.join(args.data_root, 'shanghai')
        train_folder = os.path.join(shanghai_root, 'training', 'frames')
        test_folder = os.path.join(shanghai_root, 'testing', 'frames')
        args.epochs = 10
    if not os.path.isdir(train_folder) or not os.path.isdir(test_folder):
        raise FileNotFoundError(
            f"Dataset folders not found: train={train_folder}, test={test_folder}"
        )
    img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
    exp_dir = (
        f'{args.exp_dir}_lr{learning_rate}_bs{args.batch_size}'
        f'_res{args.h}x{args.w}_c{args.features_root}_mem{args.mem_dim}'
        f'_{args.score_mode}{args.score_patch_size}'
        f'{"_gmse" + format(args.score_global_weight, "g") if args.score_global_weight else ""}'
        f'_sm{args.score_smoothing:g}'
        f'_{args.lr_scheduler}d{args.lr_decay_epochs or args.epochs}'
        f'_seed{args.manualSeed}'
    )
    log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
    print(log_dir)
    print(
        f"Model profile: input={args.h}x{args.w}, channels={args.features_root}, "
        f"memory={args.mem_dim}, AMP={device.type == 'cuda' and not args.no_amp}"
    )

    log_dir_writer = log_dir + "/" + "writer" + "/"
    writer = SummaryWriter(log_dir=log_dir_writer)
    if not os.path.exists(log_dir_writer):
        os.makedirs(log_dir_writer)

    # init model
    if args.start_epoch < args.epochs:
        feature_dim = args.features_root * 8
        En = Encoder(num_in_ch=channel_in, features_root=args.features_root)
        Den = Decoder(features_root=args.features_root, num_out_ch=channel_in, skip_ops=args.skip_ops)
        Dep = Decoder0(features_root=args.features_root, num_out_ch=channel_in, skip_ops=args.skip_ops)
        mem = Mem(args.mem_dim, feature_dim)
        mem_p = Mem(args.mem_dim, feature_dim)
        Dep.load_state_dict(Den.state_dict())
        En = En.to(device)
        mem = mem.to(device)
        mem_p = mem_p.to(device)
        Den = Den.to(device)
        Dep = Dep.to(device)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            En = nn.DataParallel(En)
            mem = nn.DataParallel(mem)
            mem_p = nn.DataParallel(mem_p)
            Den = nn.DataParallel(Den)
            Dep = nn.DataParallel(Dep)
        # init dataloader
        trans_compose = transforms.Compose([transforms.ToTensor()])
        train_dataset = Reconstruction3DDataLoader(train_folder, trans_compose,
                                                   resize_height=args.h, resize_width=args.w,
                                                   dataset=args.dataset_type,
                                                   img_extension=img_extension,
                                                   frame_cache_size=args.frame_cache_size)
        train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                            resize_height=args.h, resize_width=args.w,
                                                            dataset=args.dataset_type, jump=args.jump,
                                                            img_extension=img_extension,
                                                            frame_cache_size=args.frame_cache_size)

        train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, drop_last=True,
                                      **loader_options(args.num_workers, device.type == 'cuda'))

        train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                           shuffle=True, drop_last=True,
                                           generator=cpu_data_generator(args.manualSeed),
                                           **loader_options(args.num_workers, device.type == 'cuda'))

        test_dataset = Reconstruction3DDataLoader(test_folder, trans_compose,
                                                  resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                                  img_extension=img_extension, train=False,
                                                  frame_cache_size=args.frame_cache_size)

        test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                     shuffle=False, drop_last=False,
                                     **loader_options(args.num_workers_test, device.type == 'cuda'))

        params_En = list(En.parameters())
        params_Den = list(Den.parameters())
        params_Dep = list(Dep.parameters())
        params_mem = list(mem.parameters())
        params_mem_p = list(mem_p.parameters())
        params = params_En + params_Den + params_Dep + params_mem + params_mem_p
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        if args.lr_scheduler == 'cosine':
            decay_epochs = args.lr_decay_epochs or (args.epochs - args.start_epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: cosine_decay_multiplier(
                    epoch,
                    decay_epochs,
                    args.min_lr_ratio
                )
            )
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_decay_factor
            )
        else:
            scheduler = None
        amp_enabled = device.type == 'cuda' and not args.no_amp
        scaler = create_grad_scaler(amp_enabled)

        loss_func_mse = nn.MSELoss(reduction='none')
        best_accuracy = float('-inf')
        best_epoch = -1
        best_checkpoint = None
        final_epoch = args.epochs - 1

        sigma = args.sigma_noise ** 2
        yolo_model = None
        cifar_loader = None
        cifar_iter = None
        jump_iter = iter(train_batch_jump)
        dependency_state = {
            'yolo_model': yolo_model,
            'cifar_loader': cifar_loader,
            'cifar_iter': cifar_iter,
            'jump_loader': train_batch_jump,
            'jump_iter': jump_iter,
        }
        tic = time.time()
        for step in tqdm(range(args.start_epoch, args.epochs), ascii=True):
            tic_stage = time.time()
            En.train()
            Den.train()
            Dep.train()
            mem_p.train()
            mem.train()

            print('Training is start..')
            current_lr = optimizer.param_groups[0]['lr']
            epoch_losses = []
            pseudo_stats = {
                'normal_samples': 0,
                'pseudo_samples': 0,
                'pseudo_batches': 0,
                'gaussian': 0,
                'jump': 0,
                'object': 0,
                'shuffle': 0,
            }
            for j, imgs in enumerate(train_batch):
                loss, batch_pseudo_stats = efficient_train_step(
                    imgs=imgs,
                    models=(En, Den, Dep, mem, mem_p),
                    optimizer=optimizer,
                    scaler=scaler,
                    loss_func_mse=loss_func_mse,
                    args=args,
                    device=device,
                    channel_in=channel_in,
                    sigma=sigma,
                    dependency_state=dependency_state,
                    amp_enabled=amp_enabled
                )
                epoch_losses.append(loss.detach().item())
                pseudo_stats['normal_samples'] += batch_pseudo_stats['normal_samples']
                pseudo_stats['pseudo_samples'] += batch_pseudo_stats['pseudo_samples']
                pseudo_type = batch_pseudo_stats['pseudo_type']
                if pseudo_type is not None:
                    pseudo_stats['pseudo_batches'] += 1
                    pseudo_stats[pseudo_type] += batch_pseudo_stats['pseudo_samples']

            if args.save_interval > 0 and (
                    (step + 1) % args.save_interval == 0 or step == args.epochs - 1):
                model_dict = {
                    'En': En.state_dict(),
                    'mem': mem.state_dict(),
                    'mem_p': mem_p.state_dict(),
                    'Den': Den.state_dict(),
                    'Dep': Dep.state_dict(),
                    'epoch': step,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                }
                torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(step)))

            train_loss = float(np.mean(epoch_losses))
            writer.add_scalar("train_loss", train_loss, int(step))
            writer.add_scalar("learning_rate", current_lr, int(step))
            writer.add_scalar("pseudo/total_samples", pseudo_stats['pseudo_samples'], int(step))
            writer.add_scalar("pseudo/triggered_batches", pseudo_stats['pseudo_batches'], int(step))
            for pseudo_type in args.pseudo_types:
                writer.add_scalar(
                    f"pseudo/{pseudo_type}_samples",
                    pseudo_stats[pseudo_type],
                    int(step)
                )

            if scheduler is not None:
                scheduler.step()

            should_evaluate = (
                (step + 1) % max(1, args.eval_interval) == 0
                or step == final_epoch
            )
            if not should_evaluate:
                continue

            # test
            print(f"Evaluating model from epoch {step}...")
            labels = np.load('./frame_labels_' + args.dataset_type + '.npy', allow_pickle=True)
            videos = OrderedDict()
            videos_list = sorted(
                video for video in glob.glob(os.path.join(test_folder, '*/'))
                if not os.path.basename(os.path.normpath(video)).lower().endswith('_gt')
            )
            for video in videos_list:
                    video_name = os.path.basename(os.path.normpath(video))
                    videos[video_name] = {}
                    videos[video_name]['path'] = video
                    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
                    videos[video_name]['frame'].sort()
                    videos[video_name]['length'] = len(videos[video_name]['frame'])

            labels_list = []
            label_length = 0
            score_list = {}

            for video in sorted(videos_list):
                video_name = os.path.basename(os.path.normpath(video))
                labels_list = np.append(labels_list,
                                                labels[0][8 + label_length:videos[video_name]['length'] + label_length - 7])
                label_length += videos[video_name]['length']
                score_list[video_name] = []

            label_length = 0
            video_num = 0
            first_video = os.path.basename(os.path.normpath(videos_list[video_num]))
            label_length += videos[first_video]['length']

            En.eval()
            Den.eval()
            with torch.no_grad():
                for k, (imgs) in enumerate(test_batch):
                    if k == label_length - 15 * (video_num + 1):
                        video_num += 1
                        current_video = os.path.basename(os.path.normpath(videos_list[video_num]))
                        label_length += videos[current_video]['length']

                    imgs = imgs.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=amp_enabled):
                        fea, x2, x1, x0 = En(imgs)
                        fea_mem = mem(fea)
                        out, out1, out2, out3 = Den(fea_mem["x3_out"], x2, x1, x0)
                        frame_score, mse_imgs = reconstruction_score(
                            out[0, :, 8],
                            imgs[0, :, 8],
                            args.score_mode,
                            args.score_patch_size,
                            args.score_global_weight
                        )
                    current_video = os.path.basename(os.path.normpath(videos_list[video_num]))
                    score_list[current_video].append(frame_score)

            anomaly_score_total_list = []
            for vi, video in enumerate(sorted(videos_list)):
                video_name = os.path.basename(os.path.normpath(video))
                score = anomaly_score_list(score_list[video_name])
                anomaly_score_total_list += smooth_video_scores(score, args.score_smoothing)
            anomaly_score_total_list = np.asarray(anomaly_score_total_list)
            evaluation_labels = 1 - labels_list if args.score_mode == 'psnr' else labels_list
            accuracy = AUC(anomaly_score_total_list, np.expand_dims(evaluation_labels, 0))

            # 记录日志
            toc_stage = time.time()
            write2txt(log_dir,
                      f'model_{step}_AUC: {accuracy * 100:.2f}% | train loss: {train_loss:.4f} | '
                      f'test loss: {mse_imgs:.4f} | psnr: {psnr(mse_imgs):.4f} | '
                      f'score: {args.score_mode}({args.score_patch_size}) | '
                      f'global mse: {args.score_global_weight:g} | '
                      f'smoothing: {args.score_smoothing:g} | lr: {current_lr:.3e} | '
                      f'time: {(toc_stage - tic_stage) / 60:.2f} min')
            writer.add_scalar("auroc_img", accuracy, step)
            writer.add_scalar("test_loss", mse_imgs, step)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = step
                best_checkpoint = {
                    'En': {key: value.detach().cpu().clone()
                           for key, value in En.state_dict().items()},
                    'Den': {key: value.detach().cpu().clone()
                            for key, value in Den.state_dict().items()},
                    'Dep': {key: value.detach().cpu().clone()
                            for key, value in Dep.state_dict().items()},
                    'mem': {key: value.detach().cpu().clone()
                            for key, value in mem.state_dict().items()},
                    'mem_p': {key: value.detach().cpu().clone()
                              for key, value in mem_p.state_dict().items()},
                    'epoch': step,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'score_smoothing': args.score_smoothing,
                    'score_mode': args.score_mode,
                    'score_patch_size': args.score_patch_size,
                    'score_global_weight': args.score_global_weight,
                }

        toc = time.time()
        if best_checkpoint is None:
            raise RuntimeError("No test-set evaluation was run")
        best_ckp_path = os.path.join(log_dir, f"best_epoch_{best_epoch}.pth")
        print('total time:' + str((toc - tic) / 3600) + "h")
        print('mean time:' + str((toc - tic) / 60 / (args.epochs - args.start_epoch)) + "min")
        print('best_ckp_path: ' + best_ckp_path)

        torch.save(best_checkpoint, best_ckp_path)
        write2txt(log_dir, '-----------------------------------------')
        write2txt(log_dir, f'best_model_{best_epoch}_AUC: {best_accuracy * 100}% '
                           f'total time: {(toc - tic) / 60} min')
        return best_accuracy, best_epoch


def write2txt(filename, content):
    f = open(os.path.join(filename, 'log.txt'), 'a')
    f.write(str(content) + "\n")
    f.close()


parser = argparse.ArgumentParser(description="VAD")
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')
parser.add_argument('--eval_interval', type=int, default=1,
                    help='evaluate on the test set every N epochs')
parser.add_argument('--save_interval', type=int, default=1,
                    help='save a training checkpoint every N epochs; 0 saves only the best model')
parser.add_argument('--score_smoothing', type=float, default=32.0,
                    help='Gaussian temporal smoothing sigma for per-video anomaly scores; 0 disables it')
parser.add_argument('--score_mode', choices=['psnr', 'mse', 'max_patch'], default='max_patch',
                    help='frame anomaly score used for AUC evaluation')
parser.add_argument('--score_patch_size', type=int, default=8,
                    help='residual patch size used by max_patch scoring')
parser.add_argument('--score_global_weight', type=float, default=0.4,
                    help='global MSE weight mixed with max-patch score, from 0 to 1')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--learning_rate_ped2', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--lr_scheduler', choices=['none', 'cosine', 'step'], default='cosine',
                    help='learning-rate scheduler applied to Adam')
parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                    help='cosine scheduler minimum LR as a ratio of the initial LR')
parser.add_argument('--lr_decay_epochs', type=int, default=10,
                    help='fixed cosine decay length; 0 uses the total training epochs')
parser.add_argument('--lr_step_size', type=int, default=8,
                    help='epochs between LR reductions for the step scheduler')
parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                    help='LR multiplier used by the step scheduler')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--frame_cache_size', type=int, default=64,
                    help='number of decoded frames cached per dataset worker; 0 disables the cache')
parser.add_argument('--loss_m_weight', help='loss_m_weight', type=float, default=0.0002)
parser.add_argument('--diff_loss_weight', type=float, default=0.0002,
                    help='weight shared by the L1 and two KL feature-difference terms')
parser.add_argument('--lambda_l1', type=float, default=1.0,
                    help='internal weight of the L1 feature-difference term')
parser.add_argument('--lambda_kl_fea', type=float, default=1.0,
                    help='internal weight of the bottleneck KL-divergence term')
parser.add_argument('--lambda_kl_dec', type=float, default=1.0,
                    help='internal weight of the decoder KL-divergence term')
parser.add_argument('--disable_sparsity_loss', action='store_true',
                    help='exclude attention sparsity loss for loss ablation')
parser.add_argument('--disable_diff_loss', action='store_true',
                    help='exclude all feature-difference loss terms for loss ablation')

parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--data_root', type=str, default=r'E:\dataset',
                    help='root directory containing UCSDped2, Avenue and ShanghaiTech datasets')
parser.add_argument('--path', type=str, default='exp_log', help='directory of data')
parser.add_argument('--path_num', type=int, default=11, help='number of path')
parser.add_argument('--mem_dim', type=int, default=500, help='size of mem')
parser.add_argument('--features_root', type=int, default=16,
                    help='base channel count of the 3D autoencoder')
parser.add_argument('--sigma_noise', default=0.9, type=float, help='sigma of noise added to the iamges')
parser.add_argument('--exp_dir', type=str, default='ped2_sm20_02', help='basename of folder to save weights')
parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
parser.add_argument(
    '--skip_ops',
    nargs=3,
    choices=['none', 'concat'],
    default=['none', 'none', 'concat'],
    metavar=('LEVEL1', 'LEVEL2', 'LEVEL3'),
    help='skip connection operation for shallow, middle, and deep levels'
)
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# Miscs
parser.add_argument('--manualSeed', type=int, default=2026, help='manual random seed')

# Device options
parser.add_argument('--gpu-id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pseudo_anomaly', type=float, default=0.02,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2],
                    help='Jump for pseudo anomaly (hyperparameter s)')
parser.add_argument('--pseudo_types', nargs='+',
                    choices=['gaussian', 'jump', 'object', 'shuffle'],
                    default=['gaussian', 'jump', 'object', 'shuffle'],
                    help='enabled pseudo anomaly generators')
parser.add_argument('--no_amp', action='store_true',
                    help='disable CUDA automatic mixed precision')
parser.add_argument('--deterministic', action='store_true',default=True,
                    help='use deterministic cuDNN kernels (slower)')

args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    setup_seed(args.manualSeed, deterministic=args.deterministic)
    print(args)

    best_accuracy, best_epoch = train_test(args)
    print("best_auroc_img:", str(best_accuracy), "epoch:", str(best_epoch))
