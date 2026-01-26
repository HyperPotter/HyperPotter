import argparse
import sys
import os
import numpy as np
import torch
import pandas as pd
from torch import nn
import socket
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from data_utils import Default_Augment, Default_dataset
from model import Model
from comp_eer import compute_eer
import torch.nn.functional as F
from tqdm import tqdm

from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import random
from sklearn.metrics import roc_auc_score


from torch.utils.data.sampler import Sampler

def _force_contiguous_grads(module: nn.Module):
    for p in module.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: g.contiguous() if g is not None else g)


def gather_predictions_from_all_gpus(scores, labels, device):
    """
    Gather prediction results across all GPUs and return
    the complete dataset scores and labels.
    """
    if not dist.is_initialized():
        return scores, labels
    
    # Convert to tensors
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.int64, device=device)
    
    # Collect data length from all GPUs
    local_size = torch.tensor([len(scores)], dtype=torch.long, device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    max_size = max([s.item() for s in size_list])
    
    # Pad to the same length
    if len(scores) < max_size:
        pad_size = max_size - len(scores)
        scores_t = torch.cat([scores_t, torch.zeros(pad_size, device=device)])
        labels_t = torch.cat(
            [labels_t, torch.zeros(pad_size, dtype=torch.int64, device=device)]
        )
    
    # Gather results from all GPUs
    gathered_scores = [
        torch.zeros(max_size, device=device) for _ in range(dist.get_world_size())
    ]
    gathered_labels = [
        torch.zeros(max_size, dtype=torch.int64, device=device)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_scores, scores_t)
    dist.all_gather(gathered_labels, labels_t)
    
    # Concatenate all valid data
    all_scores = []
    all_labels = []
    for i, size in enumerate(size_list):
        all_scores.append(gathered_scores[i][:size.item()].cpu().numpy())
        all_labels.append(gathered_labels[i][:size.item()].cpu().numpy())
    
    return np.concatenate(all_scores), np.concatenate(all_labels)



def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)

class CustomBatchSampler(Sampler):
    r"""Yield a mini-batch of indices. 

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, data_length, batch_size, total):
        # build data for sampling here
        self.batch_size = batch_size
        self.data = random.sample(range(data_length), total)
        self.total = total
        
        
    def __iter__(self):
        # implement logic of sampling here
        batch = []
        for i in self.data:
            batch.append(i)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return self.total//self.batch_size


def evaluate_accuracy(data_loader, model, device, criterion, save_score_path=None):
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    labels = []
    scores = []
    file_names = []

    with torch.no_grad():
        loop = tqdm(data_loader, desc='eval', dynamic_ncols=True)
        for batch_data in loop:
            if len(batch_data) == 4:
                batch_x, batch_y, lengths, batch_file_name = batch_data
            else:
                batch_x, batch_y, batch_file_name = batch_data
                lengths = None

            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.view(-1).long().to(device)

            if lengths is not None:
                lengths = lengths.to(device)
                batch_out = model(batch_x, lengths=lengths)
            else:
                batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)
            val_loss += batch_loss.item() * batch_size

            # Probability/score for class 1 (bonafide)
            batch_score = F.softmax(batch_out, dim=1)[:, 1]
            batch_score = batch_score.detach().cpu().numpy().ravel()
            batch_y = batch_y.cpu().numpy()

            scores.append(batch_score)
            labels.append(batch_y)
            file_names.extend(batch_file_name)

            loop.set_postfix(loss=val_loss / num_total)

    # ===== Aggregate loss across GPUs (global mean loss) =====
    if dist.is_initialized():
        loss_sum_t = torch.tensor([val_loss], dtype=torch.float64, device=device)
        num_total_t = torch.tensor([num_total], dtype=torch.float64, device=device)
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_total_t, op=dist.ReduceOp.SUM)
        val_loss = (loss_sum_t / (num_total_t + 1e-12)).item()
    else:
        val_loss /= num_total

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # ===== Cross-GPU gather for scores/labels =====
    scores, labels = gather_predictions_from_all_gpus(scores, labels, device)

    # ===== Gather file names across GPUs (only needed on rank 0 for saving) =====
    if dist.is_initialized():
        gathered_file_names = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_file_names, file_names)
        if dist.get_rank() == 0:
            file_names = sum(gathered_file_names, [])
    else:
        gathered_file_names = None

    # ===== Save score file (rank 0 only) =====
    if save_score_path is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        assert len(file_names) == len(scores), "Mismatch between file_name and score counts"

        df_score = pd.DataFrame({
            "file_name": file_names,
            "label": labels.astype(int),
            "score": scores.astype(float)
        })
        df_score.to_csv(save_score_path, index=False)
        print(f"[INFO] Score file saved to: {save_score_path}")

    # ===== Metrics =====
    auc = roc_auc_score(labels, scores)

    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(bonafide_scores.shape, spoof_scores.shape)

    eer, th = compute_eer(bonafide_scores, spoof_scores)

    # Accuracy under the EER threshold
    acc = (
        np.sum(bonafide_scores > th) +
        np.sum(spoof_scores <= th)
    ) / (len(bonafide_scores) + len(spoof_scores))

    # Predicted label: 1 if score > threshold else 0
    predict_labels = (scores > th).astype(np.int64)

    # Per-class precision/recall/F1 (class 0: spoof, class 1: bonafide)
    num_classes = 2
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    for i in range(num_classes):
        tp[i] = np.sum((predict_labels == i) & (labels == i))
        fp[i] = np.sum((predict_labels == i) & (labels != i))
        fn[i] = np.sum((predict_labels != i) & (labels == i))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    macro_f1 = f1.mean().item()

    return val_loss, eer, acc, auc, f1[0].item(), f1[1].item(), macro_f1



def train_epoch(train_loader, model, lr, optim, device, criterion, bank_sync_every: int = 0):
    running_loss = 0
    real_loss_sum = 0
    fake_loss_sum = 0

    num_total = 0.0
    real_count = 0.0
    fake_count = 0.0

    model.train()

    loop = tqdm(train_loader, desc='Train', dynamic_ncols=True)

    step_i = 0
    for batch_data in loop:
        # Support variable-length inputs: check the length of returned tuple
        if len(batch_data) == 3:
            batch_x, batch_y, lengths = batch_data
        else:
            batch_x, batch_y = batch_data
            lengths = None

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.type(torch.float32).to(device)
        batch_y = batch_y.reshape(-1).type(torch.int64).to(device)

        if lengths is not None:
            lengths = lengths.to(device)

        # Set labels for prototype manager (if exists)
        if getattr(model.module, 'proto_manager', None) is not None:
            with torch.no_grad():
                model.module.proto_manager.set_batch_labels(batch_y)

        # Forward pass
        if lengths is not None:
            batch_out = model(batch_x, lengths=lengths)
        else:
            batch_out = model(batch_x)

        # Clear labels in prototype manager
        if getattr(model.module, 'proto_manager', None) is not None:
            with torch.no_grad():
                model.module.proto_manager.clear_labels()

        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        # Separate real and fake samples
        mask_real = (batch_y == 1)
        mask_fake = (batch_y == 0)

        if mask_real.sum().item() > 0:
            loss_real_batch = criterion(batch_out[mask_real], batch_y[mask_real])
            n_real = int(mask_real.sum().item())
            real_loss_sum += loss_real_batch.item() * n_real
            real_count += n_real
        else:
            loss_real_batch = None

        if mask_fake.sum().item() > 0:
            loss_fake_batch = criterion(batch_out[mask_fake], batch_y[mask_fake])
            n_fake = int(mask_fake.sum().item())
            fake_loss_sum += loss_fake_batch.item() * n_fake
            fake_count += n_fake
        else:
            loss_fake_batch = None

        # Backpropagation
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        step_i += 1

        # Synchronize prototype memory banks (optional, reduces multi-GPU drift)
        if bank_sync_every and (step_i % bank_sync_every == 0) and getattr(model.module, 'proto_manager', None) is not None:
            try:
                with torch.no_grad():
                    model.module.proto_manager.sync_banks()
            except Exception as e:
                # Synchronization failure is non-fatal; log once
                print(f"[warn] sync_banks failed: {e}")

        loop.set_postfix(loss=running_loss / num_total)
        del batch_x, batch_y

    epoch_loss_real = real_loss_sum / real_count if real_count > 0 else 0.0
    epoch_loss_fake = fake_loss_sum / fake_count if fake_count > 0 else 0.0

    running_loss /= num_total

    return running_loss, epoch_loss_real, epoch_loss_fake

def main(local_rank, args):
    # Initialize distributed training (NCCL backend)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=local_rank,
        world_size=args.world_size
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Reproducibility
    set_random_seed(args.seed, args)

    # Experiment tag / output directory
    model_tag = 'model_{}_{}'.format(args.batch_size, args.lr)
    if args.comment:
        model_tag = '{}_'.format(args.comment) + model_tag
    model_save_path = os.path.join('./models', model_tag)

    # Create output directory on rank 0 only
    if local_rank == 0 and not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # TensorBoard writer (rank 0 only)
    writer = SummaryWriter('logs/{}'.format(model_tag)) if local_rank == 0 else None

    # Build model and wrap with DDP
    model = Model(args, device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print('nb_params:', nb_params)

    # Optional: load model checkpoint
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print('Model loaded:', args.model_path)

    # Optional: load prototype/memory banks
    if args.proto_banks_path:
        model.module.proto_manager.load_banks(args.proto_banks_path)
        print('Prototype banks loaded:', args.proto_banks_path)

    # Debug option: reset memory bank
    if args.debug:
        model.module.proto_manager.bank = {}

    # Quick device sanity checks (rank 0 only)
    if local_rank == 0:
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Optimizer / loss
    ssl_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Training loss (class imbalance can be handled via weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device))

    optimizers = {'ssl': ssl_optimizer}
    optimizer = optimizers['ssl']

    # ===== Dev/Eval-only mode =====
    if args.dev:
        model.eval()
        if not args.protocols_dev_path:
            raise ValueError('Please provide --protocols_dev_path for dev mode')

        # Build dev set / loader (no shuffling)
        dev_set = Default_dataset(
            prctl_path=args.protocols_dev_path,
            transform=None,
            split=['test', '-', 'eval'],
            return_path=True
        )
        dev_sampler = DistributedSampler(
            dev_set,
            num_replicas=args.world_size,
            rank=local_rank,
            shuffle=False
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=args.eval_batch_size,
            num_workers=32,
            drop_last=True,
            sampler=dev_sampler,
            pin_memory=True,
            collate_fn=collate_fn
        )
        print('no. of evaluation trials', len(dev_set))
        del dev_set

        # For evaluation, optionally use balanced weights
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]).to(device))
        dev_loss, eer, acc, auc, f1_spoof, f1_bona, macro_f1 = evaluate_accuracy(
            dev_loader, model, device, criterion
        )

        if local_rank == 0:
            print("dataset:", args.protocols_dev_path)
            print(
                'Dev loss: {:.4f}, EER: {:.4f}, ACC: {:.4f}, AUC: {:.4f}, '
                'F1 Spoof: {:.4f}, F1 Bona: {:.4f}, Macro F1: {:.4f}'.format(
                    dev_loss, eer, acc, auc, f1_spoof, f1_bona, macro_f1
                )
            )

        sys.exit(0)

    # ===== Training data loader =====
    augmenter = Default_Augment(args=args)
    train_set = Default_dataset(
        prctl_path=args.protocols_trn_path,
        transform=augmenter,
        split='train'
    )
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=args.world_size,
        rank=local_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print('no. of training trials', len(train_set))
    del train_set

    # ===== Wild evaluation loader =====
    wild_eval_set = Default_dataset(
        prctl_path=args.protocols_wild_path,
        transform=None,
        split='-',
        max_length=args.max_length,
        return_path=True
    )
    wild_eval_sampler = DistributedSampler(
        wild_eval_set,
        num_replicas=args.world_size,
        rank=local_rank,
        shuffle=False
    )
    wild_eval_loader = DataLoader(
        wild_eval_set,
        batch_size=args.eval_batch_size,
        num_workers=32,
        sampler=wild_eval_sampler,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print('no. of wild evaluation trials', len(wild_eval_set))
    del wild_eval_set

    # ===== Training loop =====
    num_epochs = args.num_epochs
    min_wild_eer = 100

    for epoch in range(args.start_epoch, num_epochs):
        print("epoch", epoch, "start")

        # Warm-up and anneal ProtoManager gating (if enabled)
        pm = model.module.proto_manager
        pm.warm_on_train = (epoch >= args.warm_start_epoch)

        # Linear annealing for gate_tau from start to end during warm period
        if pm.warm_on_train:
            total_span = max(1, num_epochs - args.warm_start_epoch)
            progress = min(max((epoch - args.warm_start_epoch) / float(total_span), 0.0), 1.0)
            pm.gate_tau = args.gate_tau_start + (args.gate_tau_end - args.gate_tau_start) * progress
        else:
            pm.gate_tau = args.gate_tau_start

        print(f"ProtoManager warm_on_train: {pm.warm_on_train}, gate_tau: {pm.gate_tau:.4f}")

        # Enable/disable slot alignment based on epoch
        model.module.proto_manager.use_slot_alignment = (epoch >= args.warm_end_epoch)

        # Train one epoch
        running_loss, loss_real, loss_fake = train_epoch(
            train_loader,
            model,
            args.lr,
            optimizer,
            device,
            criterion,
            bank_sync_every=args.bank_sync_every
        )

        # Optional: extra bank sync after each epoch to keep replicas aligned
        with torch.no_grad():
            model.module.proto_manager.sync_banks()

        # Evaluate on the wild set
        wild_loss, wild_eer, wild_acc, wild_auc, wild_f1_spoof, wild_f1_bona, wild_macro_f1 = evaluate_accuracy(
            wild_eval_loader, model, device, criterion
        )

        if local_rank == 0:
            writer.add_scalar('train_loss', running_loss, epoch)
            writer.add_scalar('train_loss/loss_real', loss_real, epoch)
            writer.add_scalar('train_loss/loss_fake', loss_fake, epoch)
            writer.add_scalar('wild_eer', wild_eer, epoch)
            writer.add_scalar('wild_acc', wild_acc, epoch)

            print('\nep{} - loss:{} - wild:{}'.format(epoch, running_loss, wild_eer))
            with open(model_save_path + '/train_log', "a") as f:
                f.write('\nep{} - loss:{} - wild:{}'.format(epoch, running_loss, wild_eer))

            # Save best checkpoints (rank 0 only)
            if wild_eer < min_wild_eer:
                min_wild_eer = wild_eer
                if epoch > 10:
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                    banks_path = os.path.join(model_save_path, f'proto_banks_epoch_{epoch}.pt')
                    model.module.proto_manager.save_banks(banks_path)

        # Synchronize all ranks at the end of epoch
        dist.barrier()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HyperPotter Training and Evaluation')

    # API / runtime parameters
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--dev', action='store_true', default=False, help='Run in dev/evaluation-only mode')
    parser.add_argument(
        '--proto_banks_path',
        type=str,
        default=None,
        help='Path to prototype bank file (can be managed independently from model weights)'
    )
    parser.add_argument('--protocols_dev_path', type=str, default=None, help='Dev protocol file path')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')

    # Dataset paths
    parser.add_argument(
        '--protocols_trn_path',
        type=str,
        default='protocols/ASVspoof2019LA.txt',
        help='Training protocol file path'
    )
    parser.add_argument(
        '--protocols_wild_path',
        type=str,
        default='protocols/InTheWild.txt',
        help='Wild evaluation protocol file path'
    )

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=96, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch index')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--loss', type=str, default='weighted_CCE', help='Loss function name')
    parser.add_argument('--max_length', type=int, default=2000, help='Maximum input length')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')

    # Experiment tag
    parser.add_argument('--comment', type=str, default=None, help='Comment string to label outputs (avoid personal info)')

    # Prototype Memory warm-start & sync options
    parser.add_argument(
        '--warm_start_epoch',
        type=int,
        default=5,
        help='Enable train-time warm-start after this epoch (inclusive)'
    )
    parser.add_argument(
        '--warm_end_epoch',
        type=int,
        default=20,
        help='Enable slot alignment after this epoch (inclusive)'
    )
    parser.add_argument('--gate_tau_start', type=float, default=0.1, help='Gate tau at the start of scheduling')
    parser.add_argument('--gate_tau_end', type=float, default=0.0, help='Gate tau at the end of scheduling')
    parser.add_argument(
        '--bank_sync_every',
        type=int,
        default=10,
        help='Synchronize prototype banks every N steps during training (0 to disable)'
    )

    # Backend options
    parser.add_argument(
        '--cudnn-deterministic-toggle',
        action='store_false',
        default=True,
        help='Use cuDNN deterministic mode (default: true)'
    )
    parser.add_argument(
        '--cudnn-benchmark-toggle',
        action='store_true',
        default=False,
        help='Use cuDNN benchmark mode (default: false)'
    )

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5, help='Number of notch filters [default=5]')
    parser.add_argument('--minF', type=int, default=20, help='Minimum center frequency (Hz) [default=20]')
    parser.add_argument('--maxF', type=int, default=8000, help='Maximum center frequency (Hz) (< sr/2) [default=8000]')
    parser.add_argument('--minBW', type=int, default=100, help='Minimum filter bandwidth (Hz) [default=100]')
    parser.add_argument('--maxBW', type=int, default=1000, help='Maximum filter bandwidth (Hz) [default=1000]')
    parser.add_argument('--minCoeff', type=int, default=10, help='Minimum filter coefficients [default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, help='Maximum filter coefficients [default=100]')
    parser.add_argument('--minG', type=int, default=0, help='Minimum linear gain factor [default=0]')
    parser.add_argument('--maxG', type=int, default=0, help='Maximum linear gain factor [default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, help='Min gain diff between linear/non-linear [default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, help='Max gain diff between linear/non-linear [default=20]')
    parser.add_argument('--N_f', type=int, default=5, help='Nonlinearity order (N_f=1 is linear only) [default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, help='Max percentage of uniformly distributed samples (%) [default=10]')
    parser.add_argument('--g_sd', type=int, default=2, help='Gain parameter (> 0) [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, help='Minimum SNR for colored additive noise [default=10]')
    parser.add_argument('--SNRmax', type=int, default=40, help='Maximum SNR for colored additive noise [default=40]')

    # DDP parameters
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank (used by DDP)')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of processes / GPUs')

    # Ensure output directory exists
    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()

    # DDP master settings (single-node multi-GPU)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(find_free_port())

    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=args.world_size,
        join=True
    )
