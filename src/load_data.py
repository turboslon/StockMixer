"""Data loading utilities for StockMixer."""

import os

import numpy as np
from tqdm import tqdm


def load_eod_data(data_path, market_name, tickers, steps=1):
    """Load End-of-Day stock data from CSV files.

    Args:
        data_path: Path to the data directory
        market_name: Name of the market (e.g., 'NASDAQ', 'NYSE')
        tickers: List of stock tickers to load
        steps: Number of steps for ground truth calculation (default: 1)

    Returns:
        Tuple of (eod_data, masks, ground_truth, base_price)

    """
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tqdm(tickers)):
        single_eod = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_eod = single_eod[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_eod.shape)
            eod_data = np.zeros([len(tickers), single_eod.shape[0],
                                 single_eod.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_eod.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_eod.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_eod.shape[0]],
                                  dtype=np.float32)
        for row in range(single_eod.shape[0]):
            if abs(single_eod[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_eod[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_eod[row][-1] - single_eod[row - steps][-1]) / \
                    single_eod[row - steps][-1]
            for col in range(single_eod.shape[1]):
                if abs(single_eod[row][col] + 1234) < 1e-8:
                    single_eod[row][col] = 1.1
        eod_data[index, :, :] = single_eod[:, 1:]
        base_price[index, :] = single_eod[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    """Load graph relation data and compute normalized adjacency matrix.

    Args:
        relation_file: Path to the relation encoding file
        lap: Whether to return Laplacian matrix (default: False)

    Returns:
        Normalized adjacency matrix or Laplacian matrix

    """
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    """Load relation data with masking.

    Args:
        relation_file: Path to the relation encoding file

    Returns:
        Tuple of (relation_encoding, mask)

    """
    relation_encoding = np.load(relation_file)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_sfm_data(data_path, market_name, tickers):
    """Build SFM (Stock Feature Matrix) data from CSV files.

    Args:
        data_path: Path to the data directory
        market_name: Name of the market
        tickers: List of stock tickers to load

    """
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_eod = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_eod.shape)
            eod_data = np.zeros([len(tickers), single_eod.shape[0]],
                                dtype=np.float32)

        for row in range(single_eod.shape[0]):
            if abs(single_eod[row][-1] + 1234) < 1e-8:
                if row < 3:
                    for i in range(row + 1, single_eod.shape[0]):
                        if abs(single_eod[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_eod[i][-1]
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
            else:
                eod_data[index][row] = single_eod[row][-1]
    np.save(market_name + '_sfm_data', eod_data)
