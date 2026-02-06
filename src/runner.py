"""StockMixer runner for evaluating trained models on train/validate/test sets."""

import argparse
import os
import pickle

import numpy as np
import torch

from evaluator import evaluate
from model import StockMixer, get_loss
from train import load_checkpoint, get_device


def parse_args():
    """Parse command line arguments for StockMixer runner."""
    parser = argparse.ArgumentParser(description='StockMixer Model Runner')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--market', type=str, default='NASDAQ',
                        choices=['NASDAQ', 'NYSE', 'SP500'],
                        help='Market dataset to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu, or auto for automatic selection)')
    parser.add_argument('--splits', type=str, default='train,val,test',
                        help='Comma-separated list of splits to evaluate (train,val,test)')
    return parser.parse_args()


def load_data(market_name, dataset_path):
    """Load dataset for the specified market.

    Args:
        market_name: Name of the market (NASDAQ, NYSE, SP500)
        dataset_path: Path to the dataset directory

    Returns:
        Tuple of (eod_data, mask_data, gt_data, price_data, trade_dates)

    """
    if market_name == "SP500":
        data = np.load(os.path.join(dataset_path, 'SP500.npy'))
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        steps = 1
        for ticket in range(0, data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (
                    (data[ticket][row][-1] - data[ticket][row - steps][-1]) /
                    data[ticket][row - steps][-1]
                )
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
    return eod_data, mask_data, gt_data, price_data, trade_dates


def evaluate_split(model, device, eod_data, mask_data, gt_data, price_data,
                   start_index, end_index, lookback_length, steps, stock_num, alpha):
    """Evaluate model on a specific data split.

    Args:
        model: StockMixer model
        device: Device to run on
        eod_data: End-of-day data
        mask_data: Mask data
        gt_data: Ground truth data
        price_data: Price data
        start_index: Start index for the split
        end_index: End index for the split
        lookback_length: Lookback window length
        steps: Prediction steps
        stock_num: Number of stocks
        alpha: Alpha parameter for loss calculation

    Returns:
        Dictionary containing metrics and losses

    """
    model.eval()
    with torch.no_grad():
        cur_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.

        num_batches = end_index - lookback_length - steps + 1 - (start_index - lookback_length - steps + 1)

        for cur_offset in range(start_index - lookback_length - steps + 1,
                                end_index - lookback_length - steps + 1):
            # Get batch data
            seq_len = lookback_length
            mask_batch = mask_data[:, cur_offset: cur_offset + seq_len + steps]
            mask_batch = np.min(mask_batch, axis=1)

            data_batch = eod_data[:, cur_offset:cur_offset + seq_len, :]
            mask_batch = np.expand_dims(mask_batch, axis=1)
            price_batch = np.expand_dims(price_data[:, cur_offset + seq_len - 1], axis=1)
            gt_batch = np.expand_dims(gt_data[:, cur_offset + seq_len + steps - 1], axis=1)

            data_batch = torch.Tensor(data_batch).to(device)
            mask_batch = torch.Tensor(mask_batch).to(device)
            price_batch = torch.Tensor(price_batch).to(device)
            gt_batch = torch.Tensor(gt_batch).to(device)

            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                prediction, gt_batch, price_batch, mask_batch, stock_num, alpha
            )

            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()

            idx = cur_offset - (start_index - lookback_length - steps + 1)
            cur_pred[:, idx] = cur_rr[:, 0].cpu()
            cur_gt[:, idx] = gt_batch[:, 0].cpu()
            cur_mask[:, idx] = mask_batch[:, 0].cpu()

        loss = loss / num_batches if num_batches > 0 else 0
        reg_loss = reg_loss / num_batches if num_batches > 0 else 0
        rank_loss = rank_loss / num_batches if num_batches > 0 else 0

        perf = evaluate(cur_pred, cur_gt, cur_mask)

    return {
        'loss': loss,
        'reg_loss': reg_loss,
        'rank_loss': rank_loss,
        'mse': perf['mse'],
        'IC': perf['IC'],
        'RIC': perf['RIC'],
        'prec_10': perf['prec_10'],
        'sharpe5': perf['sharpe5']
    }


def print_results(split_name, results):
    """Print evaluation results for a split.

    Args:
        split_name: Name of the split (Train/Validation/Test)
        results: Dictionary containing metrics

    """
    print(f"\n{'=' * 60}")
    print(f"{split_name} Results")
    print(f"{'=' * 60}")
    print(f"Loss:       {results['loss']:.6f}")
    print(f"  Reg Loss: {results['reg_loss']:.6f}")
    print(f"  Rank Loss: {results['rank_loss']:.6f}")
    print(f"-" * 60)
    print(f"MSE:        {results['mse']:.6f}")
    print(f"IC:         {results['IC']:.6f}")
    print(f"RIC:        {results['RIC']:.6f}")
    print(f"Prec@10:    {results['prec_10']:.6f}")
    print(f"Sharpe5:    {results['sharpe5']:.6f}")
    print(f"{'=' * 60}\n")


def main():
    """Run the StockMixer model evaluator."""
    args = parse_args()

    # Device selection
    if args.device is None or args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")

    # Model parameters
    market_name = args.market
    stock_num = 1026
    lookback_length = 30
    valid_index = 756
    test_index = 1008
    fea_num = 5
    market_num = 20
    steps = 1
    alpha = 0.1
    scale_factor = 3

    # Load data
    dataset_path = '../dataset/' + market_name
    print(f"Loading data from {dataset_path}...")
    eod_data, mask_data, gt_data, price_data, trade_dates = load_data(
        market_name, dataset_path
    )
    print(f"Data loaded: {eod_data.shape}")

    # Initialize model
    model = StockMixer(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        market=market_num,
        scale=scale_factor
    ).to(device)

    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    checkpoint_info = load_checkpoint(args.model_path, model)
    print(f"Model loaded (trained for {checkpoint_info['epoch']} epochs)")
    print(f"Validation loss at checkpoint: {checkpoint_info['val_loss']:.6f}")

    # Parse splits to evaluate
    splits_to_eval = [s.strip().lower() for s in args.splits.split(',')]

    # Evaluate each split
    all_results = {}

    if 'train' in splits_to_eval:
        print("\nEvaluating on Training set...")
        train_results = evaluate_split(
            model, device, eod_data, mask_data, gt_data, price_data,
            lookback_length + steps - 1, valid_index,
            lookback_length, steps, stock_num, alpha
        )
        print_results("TRAIN", train_results)
        all_results['train'] = train_results

    if 'val' in splits_to_eval or 'validation' in splits_to_eval:
        print("\nEvaluating on Validation set...")
        val_results = evaluate_split(
            model, device, eod_data, mask_data, gt_data, price_data,
            valid_index, test_index,
            lookback_length, steps, stock_num, alpha
        )
        print_results("VALIDATION", val_results)
        all_results['validation'] = val_results

    if 'test' in splits_to_eval:
        print("\nEvaluating on Test set...")
        test_results = evaluate_split(
            model, device, eod_data, mask_data, gt_data, price_data,
            test_index, trade_dates,
            lookback_length, steps, stock_num, alpha
        )
        print_results("TEST", test_results)
        all_results['test'] = test_results

    # Print summary table
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
        print("-" * 80)

        metrics = ['loss', 'mse', 'IC', 'RIC', 'prec_10', 'sharpe5']
        for metric in metrics:
            train_val = all_results.get('train', {}).get(metric, float('nan'))
            val_val = all_results.get('validation', {}).get(metric, float('nan'))
            test_val = all_results.get('test', {}).get(metric, float('nan'))
            print(f"{metric:<15} {train_val:<15.6f} {val_val:<15.6f} {test_val:<15.6f}")

        print("=" * 80)


if __name__ == '__main__':
    main()
