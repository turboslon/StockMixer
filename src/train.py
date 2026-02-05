"""StockMixer training script."""

import argparse
import os
import pickle
import random

import numpy as np
import torch as torch

from evaluator import evaluate
from model import StockMixer, get_loss


class EarlyStopper:
    """Early stopping handler to stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """Initialize early stopper.

        Args:
            patience: Number of epochs to wait before stopping (default: 10)
            min_delta: Minimum change in validation loss to qualify as improvement (default: 0.0)
            verbose: Whether to print messages (default: True)

        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise

        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
        return False


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        val_loss: Current validation loss
        filepath: Path to save the checkpoint

    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint.

    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary containing epoch and val_loss

    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', float('inf'))
    }


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def parse_args():
    """Parse command line arguments for StockMixer training."""
    parser = argparse.ArgumentParser(description='StockMixer Training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run a single forward/backward pass to verify model works')
    parser.add_argument('--market', type=str, default='NASDAQ',
                        choices=['NASDAQ', 'NYSE', 'SP500'],
                        help='Market dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=123456789,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu, or auto for automatic selection)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--model-path', type=str, default='../models/stockmixer_best.pt',
                        help='Path to save the best model')
    parser.add_argument('--min-delta', type=float, default=0.0,
                        help='Minimum change in validation loss to qualify as improvement')
    return parser.parse_args()


def main():
    """Run the main training loop for StockMixer."""
    args = parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Device selection with priority: args.device > cuda > mps > cpu
    if args.device is None or args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")

    market_name = args.market
    stock_num = 1026
    lookback_length = 16
    epochs = 1 if args.dry_run else args.epochs
    valid_index = 756
    test_index = 1008
    fea_num = 5
    market_num = 20
    steps = 1
    learning_rate = args.learning_rate
    alpha = 0.1
    scale_factor = 3

    dataset_path = '../dataset/' + market_name
    if market_name == "SP500":
        data = np.load('../dataset/SP500/SP500.npy')
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(0, data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                       data[ticket][row - steps][-1]
    else:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)

    trade_dates = mask_data.shape[1]
    model = StockMixer(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        market=market_num,
        scale=scale_factor
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_valid_loss = np.inf
    batch_offsets = np.arange(0, valid_index, dtype=np.int32)

    # Initialize early stopper
    early_stopper = EarlyStopper(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    def validate(start_index, end_index):
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            loss = 0.
            reg_loss = 0.
            rank_loss = 0.
            for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
                data_batch, mask_batch, price_batch, gt_batch = (
                    torch.Tensor(x).to(device) for x in get_batch(cur_offset)
                )
                prediction = model(data_batch)
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                         stock_num, alpha)
                loss += cur_loss.item()
                reg_loss += cur_reg_loss.item()
                rank_loss += cur_rank_loss.item()
                cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
                cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
                cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
            loss = loss / (end_index - start_index)
            reg_loss = reg_loss / (end_index - start_index)
            rank_loss = rank_loss / (end_index - start_index)
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        return loss, reg_loss, rank_loss, cur_valid_perf

    def get_batch(offset=None):
        if offset is None:
            offset = random.randrange(0, valid_index)
        seq_len = lookback_length
        mask_batch = mask_data[:, offset: offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            eod_data[:, offset:offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

    # Dry-run mode: single forward/backward pass
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE: Testing model with single forward/backward pass")
        print("=" * 60)

        # Get a single batch
        data_batch, mask_batch, price_batch, gt_batch = (
            torch.Tensor(x).to(device) for x in get_batch(0)
        )

        print(f"Input shape: {data_batch.shape}")
        print(f"Device: {device}")
        if device.type == "mps":
            print("Note: Running on Apple Metal (MPS) for GPU acceleration")

        # Forward pass
        optimizer.zero_grad()
        prediction = model(data_batch)
        print(f"Output shape: {prediction.shape}")

        # Compute loss
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
            prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
        )
        print(f"Loss: {cur_loss.item():.6f}")

        # Backward pass
        cur_loss.backward()
        optimizer.step()

        print("=" * 60)
        print("DRY RUN COMPLETE: Model is working correctly!")
        print("=" * 60)
        return

    # Normal training mode
    for epoch in range(epochs):
        print(f"epoch{epoch + 1}##########################################################")
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        for j in range(valid_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = (
                torch.Tensor(x).to(device) for x in get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                stock_num, alpha)
            cur_loss.backward()
            optimizer.step()

            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()
        tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
        tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
        tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
        print(f'Train : loss:{tra_loss:.2e}  =  {tra_reg_loss:.2e} + alpha*{tra_rank_loss:.2e}')

        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
        print(f'Valid : loss:{val_loss:.2e}  =  {val_reg_loss:.2e} + alpha*{val_rank_loss:.2e}')

        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
        print(f'Test: loss:{test_loss:.2e}  =  {test_reg_loss:.2e} + alpha*{test_rank_loss:.2e}')

        # Save best model and check early stopping
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, args.model_path
            )
            print(f"New best model saved with validation loss: {val_loss:.6f}")

        # Check early stopping
        if early_stopper(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopper.best_loss:.6f}")
            break

        val_metrics = (
            f"mse:{val_perf['mse']:.2e}, IC:{val_perf['IC']:.2e}, "
            f"RIC:{val_perf['RIC']:.2e}, prec@10:{val_perf['prec_10']:.2e}, "
            f"SR:{val_perf['sharpe5']:.2e}"
        )
        test_metrics = (
            f"mse:{test_perf['mse']:.2e}, IC:{test_perf['IC']:.2e}, "
            f"RIC:{test_perf['RIC']:.2e}, prec@10:{test_perf['prec_10']:.2e}, "
            f"SR:{test_perf['sharpe5']:.2e}"
        )
        print('Valid performance:\n', val_metrics)
        print('Test performance:\n', test_metrics, '\n\n')


if __name__ == '__main__':
    main()
