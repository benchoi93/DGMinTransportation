import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
from scipy.stats import entropy
from itertools import combinations

# -------------------------------
# Dequantization of data
# -------------------------------
def dequantize_data(data):
    """
    Add uniform noise in [0,1) range to each column of the data to convert it to continuous.
    """
    data_continuous = data.copy().astype(float)
    for column in data_continuous.columns:
        data_continuous[column] += np.random.uniform(0, 1, size=data_continuous.shape[0])
    return data_continuous

# -------------------------------
# Data loading and preprocessing
# -------------------------------
def load_and_preprocess_data(filepath, device):
    """
    Read a CSV file, print the number of unique values in each column, and convert the data to a tensor by dequantizing it.
    """
    df = pd.read_csv(filepath, index_col=None)

    # Check the number of unique categories in each column
    num_categories = {}
    for column in df.columns:
        num_unique = df[column].nunique()
        num_categories[column] = num_unique
        print(f"Column '{column}' has {num_unique} unique categories.")

    # Dequantize the data
    df_continuous = dequantize_data(df)
    data_tensor = torch.tensor(df_continuous.values, dtype=torch.float32).to(device)

    return df, data_tensor, num_categories

# -------------------------------
# Visualization
# -------------------------------
def visualize_results(model_name, ground_truth_df, generated_df, columns, xtick_labels_list, base_font_size=16):
    ylim_list = [0.60, 0.5, 0.3, 0.6]
    sub_title_list = ['Origin Location Type', 'Activity Type', 'Mode Type', 'Destination Location Type']
    bar_width = 0.35
    offset = bar_width / 2

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']

    fig, axs = plt.subplots(1, len(columns), figsize=(20, 3))
    formatter = FormatStrFormatter('%.2f')
    fig.tight_layout(h_pad=1, w_pad=3)

    for column, ylim, sub_title, xtick_labels, ax in zip(columns, ylim_list, sub_title_list, xtick_labels_list, axs.flatten()):
        # Ground truth data
        unique_gt, counts_gt = np.unique(ground_truth_df[column], return_counts=True)
        ax.set_ylim([0, ylim])
        positions_gt = np.arange(len(unique_gt))
        ax.bar(positions_gt - offset, counts_gt / counts_gt.sum(),
               width=bar_width, color='#00008F', edgecolor='black', alpha=0.3, label='Ground Truth')

        # Generated data
        unique_gen, counts_gen = np.unique(generated_df[column], return_counts=True)
        positions_gen = np.arange(len(unique_gen))
        ax.bar(positions_gen + offset, counts_gen / counts_gen.sum(),
               width=bar_width, color='#FF4040', edgecolor='black', alpha=0.3, label='Generated')

        ax.set_xticks([int(x) for x in unique_gt])
        ax.set_xticklabels(xtick_labels, fontsize=base_font_size+4, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_yticklabels([f'{x:.2f}' for x in ax.get_yticks()], fontsize=base_font_size+4)
        ax.set_title(sub_title, fontsize=base_font_size+6)

    fig.suptitle(model_name, fontsize=base_font_size+10, y=1.15)
    fig.text(-0.03, 0.5, 'Proportion', va='center', rotation='vertical', fontsize=base_font_size+4)
    fig.legend(['Ground Truth', 'Generated'], loc='upper right', ncol=2, fontsize=base_font_size,
               bbox_to_anchor=(0.993, 1.24), bbox_transform=fig.transFigure)

    save_dir = 'img'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_discrete.png'), bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()

# -------------------------------
# Evaluation Metrics
# -------------------------------

def calculate_mse(real_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        real_data: Real data distribution
        generated_data: Generated data distribution
        
    Returns:
        float: MSE value
    """
    return np.mean((real_data - generated_data) ** 2)

def calculate_mae(real_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        real_data: Real data distribution
        generated_data: Generated data distribution
        
    Returns:
        float: MAE value
    """
    return np.mean(np.abs(real_data - generated_data))

def calculate_kld(real_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Calculate KL Divergence.
    
    Args:
        real_data: Real data distribution
        generated_data: Generated data distribution
        
    Returns:
        float: KLD value
    """
    # Add small value to prevent division by zero
    epsilon = 1e-10
    real_data = real_data + epsilon
    generated_data = generated_data + epsilon
    
    # Normalize to probability distributions
    real_dist = real_data / np.sum(real_data)
    generated_dist = generated_data / np.sum(generated_data)
    
    return entropy(real_dist, generated_dist)

def calculate_srmse(real_dist: np.ndarray, gen_dist: np.ndarray) -> float:
    """
    SRMSE = RMSE / (sum(real_counts)/N_b)
    RMSE = sqrt( (1/N_b) * Σ (gen_counts - real_counts)^2 )

    Args:
        real_dist: The real distribution (a vector summing to 1)
        gen_dist:  The generated distribution (a vector summing to 1)
    Returns:
        float: SRMSE value
    """
    # Number of bins
    N_b = len(real_dist)

    # Compute RMSE
    # Calculate the difference between the two distributions
    diff = real_dist - gen_dist
    rmse = np.sqrt(np.sum(diff ** 2)/N_b)
    
    # average probability
    avg_pi = np.sum(real_dist) / N_b
    
    # SRMSE scaling
    srmse_value = rmse / avg_pi
    
    return srmse_value

def calculate_srmse_for_columns(ground_truth_df: pd.DataFrame,
                                generated_df: pd.DataFrame,
                                cols: list) -> float:
    """
    For the given cols (e.g., two columns), compute the multi-dimensional cross distribution 
    and then calculate the SRMSE.

    Args:
        ground_truth_df: The real DataFrame
        generated_df:    The generated DataFrame
        cols:            A list of columns (usually 2) to compute SRMSE for
        
    Returns:
        float: The SRMSE value for the combination of the specified columns
    """
    # (1) Get group counts for the specified columns in the real data
    real_counts = ground_truth_df.groupby(cols).size()
    # (2) Get group counts for the specified columns in the generated data
    gen_counts = generated_df.groupby(cols).size()

    # Combine all possible category combinations (from both real and generated data) 
    # to expand the index
    unique_combos = pd.Index(list(set(real_counts.index) | set(gen_counts.index)))
    
    # Fill missing combinations with 0 via reindex
    real_counts = real_counts.reindex(unique_combos, fill_value=0).sort_index()
    gen_counts = gen_counts.reindex(unique_combos, fill_value=0).sort_index()
    
    # Convert to float
    real_counts = real_counts.values.astype(float)
    gen_counts = gen_counts.values.astype(float)
    
    # Normalize each distribution to sum to 1
    real_dist = real_counts / real_counts.sum()
    gen_dist  = gen_counts / gen_counts.sum()
    
    # Calculate SRMSE
    srmse_val = calculate_srmse(real_dist, gen_dist)
    return srmse_val


def evaluate_model(ground_truth_df, generated_df, columns):
    """
    Evaluate model performance using relative frequencies and return aggregated metric results.
    
    Args:
        ground_truth_df: Real data DataFrame
        generated_df: Generated data DataFrame
        columns: List of column names to evaluate
        
    Returns:
        tuple: Average MSE, MAE, and KLD values across all columns (computed on relative frequency distributions)
    """
    total_mse = 0
    total_mae = 0
    total_kld = 0

    for col in columns:

        unique_categories = np.union1d(ground_truth_df[col].unique(), generated_df[col].unique())
        
        gt_counts = ground_truth_df[col].value_counts().reindex(unique_categories, fill_value=0).sort_index().values.astype(float)
        gen_counts = generated_df[col].value_counts().reindex(unique_categories, fill_value=0).sort_index().values.astype(float)
        
        gt_prop = gt_counts / np.sum(gt_counts)
        gen_prop = gen_counts / np.sum(gen_counts)
        
        total_mse += calculate_mse(gt_prop, gen_prop)
        total_mae += calculate_mae(gt_prop, gen_prop)
        total_kld += calculate_kld(gt_prop, gen_prop)
    
    num_columns = len(columns)
    avg_mse = total_mse / num_columns
    avg_mae = total_mae / num_columns
    avg_kld = total_kld / num_columns

    print("\nAggregate Performance Metrics:")
    print("-" * 50)
    print(f"Average MSE: {avg_mse:.7f}")
    print(f"Average MAE: {avg_mae:.7f}")
    print(f"Average KLD: {avg_kld:.7f}")

    # Calculate for (4C2 = 6) combinations 
    two_col_combinations = list(combinations(columns, 2))
    srmse_dict = {}
    
    for combo in two_col_combinations:
        # combo ex): ('start_type', 'act_num')
        srmse_val = calculate_srmse_for_columns(ground_truth_df, generated_df, list(combo))
        srmse_dict[combo] = srmse_val
    
    print("\nSRMSE for 2-Column Combinations:")
    print("-" * 50)
    for k, v in srmse_dict.items():
        print(f"{k}: {v:.7f}")
    avg_srmse = np.mean(list(srmse_dict.values()))
    print(f"\n[Average SRMSE across all {len(two_col_combinations)} combos] = {avg_srmse:.7f}")
    
    return avg_mse, avg_mae, avg_kld, srmse_dict