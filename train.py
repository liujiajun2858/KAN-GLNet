import argparse
import os
import torch.nn.utils as utils
from block.Sopia import SophiaG
from block.lion_pytorch.lion_pytorch import Lion
from data_utils.S3DISDataLoarder_val import S3DISDataset
# from data_utils.change import  S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import matplotlib.pyplot as plt  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['茎杆', '果荚']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--epoch', default=250, type=int, help='Epochs')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer type')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='Log dir')
    parser.add_argument('--decay_rate', type=float, default=1e-2, help='Weight decay') #1e-4
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number')
    # 添加余弦退火最小学习率参数
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--val_area', type=int, default=2, help='Validation area')
    parser.add_argument('--test_area', type=int, default=3, help='Test area (not used)')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 新增绘图函数
    def plot_metrics(log_dir):
        plt.figure(figsize=(18, 6))

        # Loss曲线
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.legend()

        # Accuracy曲线
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.legend()

        # mIoU曲线
        plt.subplot(1, 3, 3)
        plt.plot(train_mious, label='Train')
        plt.plot(val_mious, label='Validation')
        plt.title('mIoU Curve')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
        plt.close()

    '''环境设置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''目录创建'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''日志配置'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''参数配置'''
    root = 'data/10/'  # duiqi4096_10plus是林慧增强方式
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    '''数据集加载'''
    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT,
                                 test_area=args.test_area, val_area=args.val_area,
                                 block_size=1, sample_rate=1.0, transform=None)

    print("start loading val data ...")
    VAL_DATASET = S3DISDataset(split='val', data_root=root, num_point=NUM_POINT,
                               test_area=args.test_area, val_area=args.val_area,
                               block_size=0.5, sample_rate=1.0, transform=None)

    '''数据加载器'''
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=True
    )
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string(f"Training samples: {len(TRAIN_DATASET)}")
    log_string(f"Validation samples: {len(VAL_DATASET)}")

    '''模型加载'''
    MODEL = importlib.import_module(args.model)
    shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    # 参数计算添加位置 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_string(f"模型总参数量: {total_params / 1e6:.2f}M")
    log_string(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 参数计算添加位置 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    def weights_init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            torch.nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(f'{experiment_dir}/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Loaded pretrained model')
    except:
        log_string('Training from scratch...')
        start_epoch = 0
        classifier.apply(weights_init)

    '''优化器配置'''
    optimizer_config = {
        'Adam': torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        ),
        # 修正AdamW - 添加正确的betas和eps参数
        'AdamW': torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),  # 添加缺失的betas参数
            eps=1e-08,  # 添加缺失的eps参数
            weight_decay=args.decay_rate
        ),
        'Lion': Lion(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.decay_rate
        ),
        'SophiaG': SophiaG(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.965, 0.99),
            weight_decay=args.decay_rate
        )
    }
    optimizer = optimizer_config.get(args.optimizer, torch.optim.SGD(
        classifier.parameters(),
        lr=args.learning_rate,
        momentum=0.9
    ))

    '''添加余弦退火调度器 - 使用新的min_lr参数'''
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,  # 总周期数
        eta_min=args.min_lr  # 使用参数传入的最小学习率
    )

    '''加载调度器状态（如果存在）'''
    try:
        checkpoint = torch.load(f'{experiment_dir}/checkpoints/best_model.pth')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        log_string('Loaded scheduler state')
    except:
        log_string('Initializing new scheduler')

    '''训练参数配置'''

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.momentum = momentum

    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = 20  # 改为固定值20，因为已移除args.step_size

    '''训练循环'''
    global_epoch = 0
    best_val_iou = 0

    # 新增指标存储
    train_losses = []
    train_accs = []
    train_mious = []
    val_losses = []
    val_accs = []
    val_mious = []

    # 记录学习率变化
    lr_history = []

    def evaluate_model(dataloader, desc="Evaluating"):
        classifier.eval()
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        with torch.no_grad():
            for i, (points, target) in tqdm(enumerate(dataloader), desc=desc, total=len(dataloader)):
                points = points.float().cuda()
                target = target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1)
                loss = criterion(seg_pred, batch_label, trans_feat, weights)
                loss_sum += loss.item()

                pred_val = seg_pred.argmax(dim=1).cpu().numpy()
                batch_label_np = batch_label.cpu().numpy()

                correct = np.sum(pred_val == batch_label_np)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                tmp, _ = np.histogram(batch_label_np, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label_np == l)
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label_np == l))
                    total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label_np == l))

        labelweights /= np.sum(labelweights)
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
        accuracy = total_correct / float(total_seen)
        class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))

        return {
            "loss": loss_sum / len(dataloader),
            "mIoU": mIoU,
            "accuracy": accuracy,
            "class_acc": class_acc,
            "iou_per_class": [total_correct_class[l] / total_iou_deno_class[l] for l in range(NUM_CLASSES)]
        }

    for epoch in range(start_epoch, args.epoch):
        log_string(f'**** Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}) ​****')

        '''获取当前学习率'''
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        log_string(f'Learning rate: {current_lr:.8f}')

        '''BN动量调整'''
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        momentum = max(momentum, 0.01)
        classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        log_string(f'BN momentum: {momentum:.4f}')

        '''训练步骤'''
        classifier.train()
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        with tqdm(trainDataLoader, desc=f"Epoch {epoch + 1}", unit="batch") as tepoch:
            for i, (points, target) in enumerate(tepoch):
                optimizer.zero_grad()

                points = points.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points).float().cuda()
                target = target.long().cuda()

                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.reshape(-1, NUM_CLASSES)

                loss = criterion(seg_pred, target.view(-1), trans_feat, weights)
                loss.backward()

                utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()

                pred_choice = seg_pred.argmax(dim=1).cpu().numpy()
                target_np = target.view(-1).cpu().numpy()
                correct = np.sum(pred_choice == target_np)
                total_correct += correct
                total_seen += BATCH_SIZE * NUM_POINT
                loss_sum += loss.item()

                # 新增mIoU统计
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(target_np == l)
                    total_correct_class[l] += np.sum((pred_choice == l) & (target_np == l))
                    total_iou_deno_class[l] += np.sum((pred_choice == l) | (target_np == l))

                tepoch.set_postfix(loss=loss.item())

        # 计算训练指标
        train_loss = loss_sum / len(trainDataLoader)
        train_acc = total_correct / float(total_seen)
        train_miou = np.mean(
            np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_mious.append(train_miou)

        log_string(f'Train Loss: {train_loss:.4f}')
        log_string(f'Train Accuracy: {train_acc:.4f}')
        log_string(f'Train mIoU: {train_miou:.4f}')

        '''验证阶段'''
        val_metrics = evaluate_model(valDataLoader, "Validating")
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["accuracy"])
        val_mious.append(val_metrics["mIoU"])

        log_string('\n**** VALIDATION RESULTS ​****')
        log_string(f'Val Loss: {val_metrics["loss"]:.4f}')
        log_string(f'Val mIoU: {val_metrics["mIoU"]:.4f}')
        log_string(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        log_string('Val Class IoU:')
        for l in range(NUM_CLASSES):
            log_string(f'{seg_label_to_cat[l]:<8}: {val_metrics["iou_per_class"][l]:.4f}')

        '''模型保存'''
        if epoch % 20 == 0:
            savepath = f'{checkpoints_dir}/model_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()  # 保存调度器状态
            }, savepath)
            log_string(f'Saved model at {savepath}')

        if val_metrics["mIoU"] >= best_val_iou:
            best_val_iou = val_metrics["mIoU"]
            savepath = f'{checkpoints_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'val_mIoU': best_val_iou,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()  # 保存调度器状态
            }, savepath)
            log_string(f'Saved BEST model at {savepath} with val mIoU {best_val_iou:.4f}')
        log_string(f'当前best_model: val mIoU {best_val_iou:.4f}')

        # 更新学习率调度器
        scheduler.step()

        # 新增学习率曲线绘制
        if epoch % 10 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(lr_history, 'o-', label='Learning Rate')
            plt.title('Cosine Annealing Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'learning_rate_curve.png'))
            plt.close()

        # 新增实时可视化
        plot_metrics(log_dir)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)


