"""StockMixer Streamlit App for evaluating trained models."""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch

from evaluator import evaluate
from model import StockMixer, get_loss
from train import load_checkpoint, get_device


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
        Dictionary containing metrics, losses, predictions and ground truth

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
        'sharpe5': perf['sharpe5'],
        'sign_accuracy': perf['sign_accuracy'],
        'confusion_matrix': perf['confusion_matrix'],
        'predictions': cur_pred,
        'ground_truth': cur_gt,
        'mask': cur_mask
    }


def main():
    """Run the StockMixer Streamlit app."""
    st.set_page_config(
        page_title="StockMixer Evaluation",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 StockMixer Model Evaluation Dashboard")
    st.markdown("Evaluate trained StockMixer models on train/validation/test sets")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        model_path = st.text_input(
            "Model Path",
            value="../models/stockmixer_best.pt",
            help="Path to the trained model checkpoint"
        )

        market = st.selectbox(
            "Market",
            options=["NASDAQ", "NYSE", "SP500"],
            index=0,
            help="Select the market dataset to use"
        )

        device_option = st.selectbox(
            "Device",
            options=["auto", "cuda", "mps", "cpu"],
            index=0,
            help="Device to use for inference"
        )

        st.divider()

        run_evaluation = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    # Main content
    if run_evaluation:
        with st.spinner("Loading model and data..."):
            # Device selection
            if device_option == "auto":
                device = get_device()
            else:
                device = torch.device(device_option)
                st.info(f"Using device: {device}")

            # Model parameters
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
            dataset_path = '../dataset/' + market
            try:
                eod_data, mask_data, gt_data, price_data, trade_dates = load_data(
                    market, dataset_path
                )
                st.success(f"Data loaded: {eod_data.shape}")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

            # Initialize model
            model = StockMixer(
                stocks=stock_num,
                time_steps=lookback_length,
                channels=fea_num,
                market=market_num,
                scale=scale_factor
            ).to(device)

            # Load checkpoint
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}")
                return

            try:
                checkpoint_info = load_checkpoint(model_path, model)
                st.success(f"Model loaded (trained for {checkpoint_info['epoch']} epochs)")
                st.info(f"Validation loss at checkpoint: {checkpoint_info['val_loss']:.6f}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

        # Evaluate all splits
        with st.spinner("Evaluating on all splits..."):
            all_results = {}

            # Train split
            train_results = evaluate_split(
                model, device, eod_data, mask_data, gt_data, price_data,
                lookback_length + steps - 1, valid_index,
                lookback_length, steps, stock_num, alpha
            )
            all_results['train'] = train_results

            # Validation split
            val_results = evaluate_split(
                model, device, eod_data, mask_data, gt_data, price_data,
                valid_index, test_index,
                lookback_length, steps, stock_num, alpha
            )
            all_results['validation'] = val_results

            # Test split
            test_results = evaluate_split(
                model, device, eod_data, mask_data, gt_data, price_data,
                test_index, trade_dates,
                lookback_length, steps, stock_num, alpha
            )
            all_results['test'] = test_results

        # Create tabs for different views (removed scatter plots tab for performance)
        tab1, tab2 = st.tabs(["📊 Metrics Table", "📉 Single Stock Line Chart"])

        with tab1:
            st.header("Evaluation Metrics")

            # Create metrics DataFrame
            metrics_data = []
            metrics = ['loss', 'reg_loss', 'rank_loss', 'mse', 'IC', 'RIC', 'prec_10', 'sharpe5', 'sign_accuracy']

            for metric in metrics:
                metrics_data.append({
                    'Metric': metric,
                    'Train': all_results['train'][metric],
                    'Validation': all_results['validation'][metric],
                    'Test': all_results['test'][metric]
                })

            metrics_df = pd.DataFrame(metrics_data)

            # Format the dataframe - sign_accuracy as percentage, others as float
            def get_format_func(col_name):
                def format_func(x):
                    metric_row = metrics_df[metrics_df[col_name] == x]
                    if not metric_row.empty and metric_row['Metric'].iloc[0] == 'sign_accuracy':
                        return f"{x:.2f}%"
                    return f"{x:.6f}"
                return format_func

            st.dataframe(
                metrics_df.style.format({
                    'Train': get_format_func('Train'),
                    'Validation': get_format_func('Validation'),
                    'Test': get_format_func('Test')
                }),
                use_container_width=True,
                hide_index=True
            )

            # Confusion Matrix Section
            st.subheader("Confusion Matrix (Sign-based)")
            st.markdown("Classification based on sign of predictions vs actual values (positive vs negative)")

            cm_cols = st.columns(3)
            splits = ['train', 'validation', 'test']
            split_labels = ['Train', 'Validation', 'Test']

            for col, split, label in zip(cm_cols, splits, split_labels):
                with col:
                    cm = all_results[split]['confusion_matrix']
                    sign_acc = all_results[split]['sign_accuracy']

                    st.markdown(f"**{label}**")

                    # Create confusion matrix as a small table
                    cm_data = {
                        '': ['Pred Positive', 'Pred Negative'],
                        'Actual Positive': [cm['TP'], cm['FN']],
                        'Actual Negative': [cm['FP'], cm['TN']]
                    }
                    cm_df = pd.DataFrame(cm_data)
                    st.dataframe(cm_df, use_container_width=True, hide_index=True)
                    st.metric("Sign Accuracy", f"{sign_acc:.2f}%")

            # Also show as a summary table
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Train MSE", f"{train_results['mse']:.6f}")
                st.metric("Train IC", f"{train_results['IC']:.6f}")
                st.metric("Train RIC", f"{train_results['RIC']:.6f}")
                st.metric("Train Sign Acc", f"{train_results['sign_accuracy']:.2f}%")

            with col2:
                st.metric("Val MSE", f"{val_results['mse']:.6f}")
                st.metric("Val IC", f"{val_results['IC']:.6f}")
                st.metric("Val RIC", f"{val_results['RIC']:.6f}")
                st.metric("Val Sign Acc", f"{val_results['sign_accuracy']:.2f}%")

            with col3:
                st.metric("Test MSE", f"{test_results['mse']:.6f}")
                st.metric("Test IC", f"{test_results['IC']:.6f}")
                st.metric("Test RIC", f"{test_results['RIC']:.6f}")
                st.metric("Test Sign Acc", f"{test_results['sign_accuracy']:.2f}%")

        with tab2:
            st.header("Single Stock Time Series (Test Data)")
            st.markdown("View predicted vs actual values over time for a single stock.")

            # Stock selector
            stock_index = st.number_input(
                "Stock Index",
                min_value=0,
                max_value=stock_num - 1,
                value=0,
                help=f"Select a stock index (0 to {stock_num - 1})"
            )

            # Get data for selected stock
            stock_pred = test_results['predictions'][stock_index, :]
            stock_gt = test_results['ground_truth'][stock_index, :]
            stock_mask = test_results['mask'][stock_index, :]

            # Create time series DataFrame
            time_idx = np.arange(len(stock_pred))
            stock_df = pd.DataFrame({
                'Time': time_idx,
                'Predicted': stock_pred,
                'Actual': stock_gt,
                'Mask': stock_mask
            })

            # Filter only valid (masked) data points
            stock_df_valid = stock_df[stock_df['Mask'] > 0.5].copy()

            if len(stock_df_valid) > 0:
                st.line_chart(
                    stock_df_valid.set_index('Time')[['Predicted', 'Actual']],
                    use_container_width=True
                )

                # Show statistics for this stock
                col1, col2, col3 = st.columns(3)
                with col1:
                    mse = np.mean((stock_df_valid['Predicted'] - stock_df_valid['Actual']) ** 2)
                    st.metric("MSE", f"{mse:.6f}")
                with col2:
                    correlation = stock_df_valid['Predicted'].corr(stock_df_valid['Actual'])
                    st.metric("Correlation", f"{correlation:.6f}")
                with col3:
                    st.metric("Valid Points", f"{len(stock_df_valid)}")
            else:
                st.warning("No valid data points for this stock (mask filters out all data).")

    else:
        # Initial state - show instructions
        st.info("👈 Configure the model path and market in the sidebar, then click 'Run Evaluation' to start.")

        st.markdown("""
        ### About This Dashboard

        This app evaluates a trained StockMixer model and provides:

        1. **Metrics Table**: Compare train/validation/test metrics including:
            - Loss (total, regression, ranking)
            - MSE (Mean Squared Error)
            - IC (Information Coefficient)
            - RIC (Rank IC)
            - Prec@10 (Precision at 10)
            - Sharpe5 (Sharpe ratio for top 5)
            - Confusion Matrix (sign-based classification)

        2. **Single Stock Line Chart**: Examine predictions over time for individual stocks
        """)


if __name__ == '__main__':
    main()
