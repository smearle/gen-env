
import copy
import glob
from itertools import product
import json
import os
from pathlib import Path
from typing import Iterable

from gen_env.configs.config import EvoConfig, ILConfig, SweepConfig
from gen_env.utils import init_config
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import yaml

from plot import load_tensorboard_logs
from utils import init_il_config, init_rl_config

# from utils import get_sweep_conf_path, init_config, load_sweep_hypers, write_sweep_confs


CROSS_EVAL_DIR = 'cross_eval'

# The index of the metric in the column MultiIndex. When 0, the metric will go on top. (This is good when we are 
# are sweeping over eval metrics and want to show only one metric to save space.) Otherwise, should be -1.
METRIC_COL_TPL_IDX = 0

table_name_remaps = {
    'val_mean': 'val. mean'
}


# Function to bold the maximum value in a column for LaTeX
def format_num(s):
    # Return if not a number
    if not np.issubdtype(s.dtype, np.number):
        return s
    is_pct = False
    # Check if the header of the row
    if is_loss_column(s.name):
        is_pct = True
        s_best = s.min()

    else:
        s_best = s.max()

    col = []

    for v in s:
        if is_pct:
            v_frmt = f'{v:.2%}'
            v_frmt = v_frmt.replace('%', '\\%')
        else:
            v_frmt = f'{v:.2f}'
        if v == s_best:
            v_frmt = f'\\textbf{{{v_frmt}}}'
        col.append(v_frmt)
    
    return col


def is_loss_column(col):
    if isinstance(col, str) and 'loss' in col:
        return True
    elif isinstance(col, tuple) and 'loss' in col[METRIC_COL_TPL_IDX]:
        return True
    return False


def replace_underscores(s):
    return s.replace('_', ' ')


def process_col_str(s):
    if isinstance(s, str):
        if s in table_name_remaps:
            s = table_name_remaps[s]
        else:
            s = replace_underscores(s)
    return s


# Function to replace underscores with spaces in a string
def process_col_tpls(t):
    if isinstance(t, str):
        return process_col_str(t)
    new_s = []
    for s in t:
        s = process_col_str(s)
        new_s.append(s)
    return tuple(new_s)


def clean_df_strings(df):

    # Replace underscores in index names
    if df.index.names:
        # df.index.names = [replace_underscores(name) if name is not None else None for name in df.index.names]
        new_names = []
        for name in df.index.names:
            if name is None:
                continue
            if name in table_name_remaps:
                new_names.append(table_name_remaps[name])
            else:
                new_names.append(replace_underscores(name))
        df.index.names = new_names

    if df.columns.names:
        # df.columns.names = [replace_underscores(name) if name is not None else None for name in df.columns.names]
        new_names = []
        for name in df.columns.names:
            if name is None:
                new_names.append(name)
            elif name in table_name_remaps:
                new_names.append(table_name_remaps[name])
            else:
                new_names.append(replace_underscores(name))
        df.columns.names = new_names

    # Replace underscores in index labels for each level of the MultiIndex
    # for level in range(df.index.nlevels):
    #     df.index = df.index.set_levels([df.index.levels[level].map(replace_underscores)], level=level)

    # Replace underscores in column names
    df.columns = df.columns.map(process_col_tpls)

    return df


def cross_eval_basic(name: str, sweep_configs: Iterable[SweepConfig], hypers, eval_hypers={}, algo='il'):
    os.makedirs(os.path.join(CROSS_EVAL_DIR, name), exist_ok=True)
    stats_to_exclude = ['train_returns', 'val_returns', 'test_returns']
    if algo == 'il':
        log_dir_attr = '_log_dir_il'
    else:
        log_dir_attr = '_log_dir_rl'
    _metrics_to_keep = ['train_mean', 'val_mean', 'test_mean']

    # Save the eval hypers to the cross_eval directory, so that we know of any special eval hyperparameters that were
    # applied during eval.
    # with open(os.path.join(CROSS_EVAL_DIR, name, "eval_hypers.yaml"), 'w') as f:
    #     yaml.dump(eval_hypers, f)

    eval_hyper_ks = [k for k in eval_hypers]
    eval_hyper_combos = list(product(*[eval_hypers[k] for k in eval_hypers]))

    eval_sweep_name = ('eval_' + '_'.join(k.strip('eval_') for k, v in eval_hypers.items() if len(v) > 1 and k != 'metrics_to_keep') if 
                        len(eval_hypers) > 0 else '')


    col_headers = [k for k in eval_hyper_ks]
    col_headers.insert(METRIC_COL_TPL_IDX, '')
    col_indices = set({})

    row_headers = [tuple(v) if isinstance(v, list) else v for v in hypers]
    row_indices = []
    row_vals = []

    # Create a dataframe with basic stats for each experiment
    basic_stats_df = {}
    # for exp_dir, stats in basic_stats.items():
    for sc in sweep_configs:

        sweep_eval_configs = [copy.deepcopy(sc)]

        # Do this so that we can get the correct stats file depending on eval parameters
        # sc = init_config_for_eval(sc)

        # For each train config, also sweep over eval params to get all the relevant stats
        for eval_hyper_combo in eval_hyper_combos:
            new_sweep_eval_configs = copy.deepcopy(sweep_eval_configs)
            for sec in sweep_eval_configs:
                new_sec = copy.deepcopy(sec)
                for k, v in zip(eval_hyper_ks, eval_hyper_combo):
                    setattr(new_sec, k, v)
                new_sweep_eval_configs.append(new_sec)
            sweep_eval_configs = new_sweep_eval_configs
        
        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
        row_indices.append(row_tpl)
        
        vals = {}
        for sec in sweep_eval_configs:
            sec_col_tpl = [getattr(sec, k) for k in eval_hyper_ks]
            sc_log_dir = getattr(sc, log_dir_attr)
            sc_stats_path = os.path.join(f'{sc_log_dir}', 
                            'nn_stats.json')
            if not os.path.isfile(sc_stats_path):
                continue
            sc_stats = json.load(open(sc_stats_path))
            # HACK to combine val and test sets
            test_returns = sc_stats['val_returns'] + sc_stats['test_returns']
            # sc_stats.pop('val_mean')
            sc_stats['test_mean'] = np.mean(test_returns)
            # END HACK

            [sc_stats.pop(k) for k in stats_to_exclude]
            for k, v in sc_stats.items():
                col_tpl = copy.deepcopy(sec_col_tpl)
                col_tpl.insert(METRIC_COL_TPL_IDX, k)
                col_tpl = tuple(col_tpl)
                col_indices.add(col_tpl)
                vals[col_tpl] = v
        row_vals.append(vals)

    col_index = pd.MultiIndex.from_tuples(col_indices, names=col_headers)
    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    basic_stats_df = pd.DataFrame(row_vals, index=row_index, columns=col_index)

    # Sort columns
    basic_stats_df = basic_stats_df.sort_index(axis=1, ascending=False)
    
    # Save the dataframe to a csv
    # os.makedirs(CROSS_EVAL_DIR, exist_ok=True)
    # basic_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "basic_stats.csv")) 

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats.md"), 'w') as f:
        f.write(basic_stats_df.to_markdown())

    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats.tex"), 'w') as f:
    #     f.write(basic_stats_df.to_latex())

    seed_name = f'{algo}_seed'
    is_multi_seed = seed_name in hypers

    # Calculate stats over seeds if applicable
    if is_multi_seed:
        # Step 1: Calculate mean and standard deviation
        group_row_indices = [col for col in basic_stats_df.index.names if col != seed_name]

        basic_stats_mean_df = basic_stats_df.groupby(group_row_indices).mean()
        basic_stats_std_df = basic_stats_df.groupby(group_row_indices).std()

        # Step 2: Create a new DataFrame with the formatted "mean +/- std%" strings
        # Initialize an empty DataFrame with the same index and columns
        meanstd_df = pd.DataFrame(index=basic_stats_mean_df.index, columns=basic_stats_mean_df.columns)

    else:
        basic_stats_mean_df = basic_stats_df
        meanstd_df = pd.DataFrame(index=basic_stats_mean_df.index, columns=basic_stats_mean_df.columns)

    # Iterate over each cell to format
    for col in basic_stats_mean_df.columns:
        if is_loss_column(col):
            is_pct = True
            m_best = basic_stats_mean_df[col].min()
        else:
            is_pct = False
            m_best = basic_stats_mean_df[col].max()

        for idx in basic_stats_mean_df.index:
            mean = basic_stats_mean_df.at[idx, col]

            if is_multi_seed:
                std = basic_stats_std_df.at[idx, col]

            if is_pct:
                mean_frmt = f'{mean:.2%}'
                mean_frmt = mean_frmt.replace('%', '\\%')
                if is_multi_seed:
                    std_frmt = f'{std:.2%}'
                    std_frmt = std_frmt.replace('%', '\\%')
            else:
                mean_frmt = f'{mean:.2f}'
                if is_multi_seed:
                    std_frmt = f'{std:.2f}'
            if mean == m_best:
                mean_frmt = f'\\textbf{{{mean_frmt}}}'
                if is_multi_seed:
                    std_frmt = f'\\textbf{{{std_frmt}}}'
            
            if is_multi_seed:
                meanstd_df.at[idx, col] = f'{mean_frmt} ± {std_frmt}'
            else:
                meanstd_df.at[idx, col] = mean_frmt

    # Note: If you want the std as a percentage of the mean, replace the formatting line with:
    # formatted_df.loc[idx, col] = f"{mean:.2f} +/- {std/mean*100:.2f}%
    basic_stats_mean_df = meanstd_df

    # Save the dataframe to a csv
    # basic_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "basic_stats_mean.csv"))
    
    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_basic_stats_mean.md"), 'w') as f:
        f.write(basic_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats_mean.tex"), 'w') as f:
    #     f.write(basic_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    row_levels_to_drop = \
        [level for level in basic_stats_mean_df.index.names if 
         basic_stats_mean_df.index.get_level_values(level).nunique() == 1]

    # Drop these rows
    basic_stats_concise_df = basic_stats_mean_df.droplevel(row_levels_to_drop)

    # Similarly, remove all column indices that have the same value across all columns
    col_levels_to_drop = \
        [level for level in basic_stats_mean_df.columns.names if
            basic_stats_mean_df.columns.get_level_values(level).nunique() == 1]
    
    # Drop these columns
    basic_stats_concise_df = basic_stats_concise_df.droplevel(col_levels_to_drop, axis=1)
    basic_stats_concise_df = basic_stats_concise_df[['train_mean', 'val_mean', 'test_mean']]

    # Drop the `n_parameters` `n_eval_eps` metrics, and others if `metrics_to_keep` is specified
    for col_tpl in basic_stats_concise_df.columns:
        if isinstance(col_tpl, str):
            metric_str = col_tpl
        else:
            metric_str = col_tpl[METRIC_COL_TPL_IDX]
        if metric_str == 'n_parameters' or metric_str == 'n_eval_eps':
            basic_stats_concise_df = basic_stats_concise_df.drop(columns=col_tpl)
        elif _metrics_to_keep is not None and metric_str not in _metrics_to_keep:
            basic_stats_concise_df = basic_stats_concise_df.drop(columns=col_tpl)

    # Save the dataframe to a csv
    # basic_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "basic_stats_concise.csv"))

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_basic_stats_concise.md"), 'w') as f:
        f.write(basic_stats_concise_df.to_markdown())

    # Bold the maximum value in each column
    # styled_basic_stats_concise_df = basic_stats_concise_df.apply(format_num)

    styled_basic_stats_concise_df = clean_df_strings(basic_stats_concise_df)

    latex_str = styled_basic_stats_concise_df.to_latex(
        multicolumn_format='c',
    )
    latex_str_lines = latex_str.split('\n')
    # Add `\centering` to the beginning of the table
    latex_str_lines.insert(0, '\\adjustbox{max width=\\textwidth}{%')
    latex_str_lines.insert(0, '\\centering')
    n_col_header_rows = len(styled_basic_stats_concise_df.columns.names)
    i = 4 + n_col_header_rows
    latex_str_lines.insert(i, '\\toprule')
    # Add `\label` to the end of the table
    latex_str_lines.append(f'\\label{{tab:{name}_{eval_sweep_name}}}')
    latex_str_lines.append('}')
    latex_str = '\n'.join(latex_str_lines)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{name}_{eval_sweep_name}.tex"), 'w') as f:
        f.write(latex_str)

    print(f"Basic stats for {name} saved to {CROSS_EVAL_DIR}/{name}.")


        
def cross_eval_misc(name: str, sweep_configs: Iterable[SweepConfig], hypers, algo='il'):
    log_dir_attr = f'_log_dir_{algo}'
    seed_name = f'{algo}_seed'

    # Create a dataframe with miscellaneous stats for each experiment
    row_headers = hypers

    # Get names of all training curve CSVs
    dummy_cfg = sweep_configs[0]
    csv_paths = glob.glob(os.path.join(getattr(dummy_cfg, log_dir_attr), '*.csv'))
    key_names = [csv_path.split(os.sep)[-1].removesuffix('.csv') for csv_path in csv_paths]
    # csv_paths = glob.glob(os.path.join(getattr(dummy_cfg, log_dir_attr), '*.csv'))
    # key_names = [csv_path.split(os.sep)[-1].removesuffix('.csv') for csv_path in csv_paths]

    # Load all tensorboard log files
    
    event_files = []
    # Assuming all keys will be present in the first experiment (fingers crossed)
    for sc in sweep_configs[0:1]:
        log_dir = getattr(sc, log_dir_attr)
        event_files.extend(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    
    key_names = set()
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        key_names.update(ea.scalars.Keys())

    exp_logs = [load_tensorboard_logs(getattr(sc, log_dir_attr), key_names) for sc in sweep_configs]
    
    key_names = list(key_names)

    stat_name_to_row_vals_curves = {}
    stat_name_to_all_timesteps = {}

    def interpolate_returns(ep_returns, timesteps, all_timesteps):
        # Group by timesteps and take the mean for duplicate values
        ep_returns = pd.Series(ep_returns).groupby(timesteps).mean()
        timesteps = np.unique(timesteps)
        
        # Create a Series with the index set to the unique timesteps of the ep_returns
        indexed_returns = pd.Series(ep_returns.values, index=timesteps)
        
        # Reindex the series to include all timesteps, introducing NaNs for missing values
        indexed_returns = indexed_returns.reindex(all_timesteps)
        
        # Interpolate missing values, ensuring forward fill to handle right edge
        interpolated_returns = indexed_returns.interpolate(method='linear', limit_direction='backward', axis=0)
        
        return interpolated_returns


    for k_i, key_name in enumerate(key_names):
        # Create a list of lists to show curves of metrics (e.g. reward) over the 
        # course of training (i.e. as would be logged by tensorboard)
        row_indices = []
        row_vals_curves = []
        all_timesteps = []
        step_col_name = 'steps'

        for sc_i, sc in enumerate(sweep_configs):
            sc: ILConfig
            exp_dir = getattr(sc, log_dir_attr)
            # exp_dir = sc._log_dir_il
            # Load the `progress.csv`

            stepss, valss = exp_logs[sc_i]
            steps, vals = stepss[k_i], valss[k_i]

            # train_metrics = pd.read_csv(csv_path, index_col=0)
            # train_metrics = train_metrics.sort_values(by=step_col_name, ascending=True)

            # misc_stats_path = os.path.join(exp_dir, 'misc_stats.json')
            # if os.path.exists(misc_stats_path):
            #     sc_stats = json.load(open(f'{exp_dir}/misc_stats.json'))
            # else:
            # max_timestep = train_metrics[step_col_name].max()
            max_timestep = steps[-1]
            # sc_stats = {'n_timesteps_trained': max_timestep}

            row_tpl = tuple(getattr(sc, k) for k in row_headers)
            row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
            row_indices.append(row_tpl)
            
            # vals = {}
            # for k, v in sc_stats.items():
                # vals[k] = v

            # row_vals.append(vals)
            
            row_vals_curves.append(vals)
            all_timesteps.append(steps)

        all_unique_timesteps = np.sort(np.unique(np.concatenate(all_timesteps)))

        row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
        # curve_stats_df = pd.DataFrame(row_vals, index=row_index)
        # stat_name_to_row_vals_curves[key_name] = row_vals_curves

        interped_row_vals_curves = []
        for i, sc in enumerate(sweep_configs):
            sc_timesteps = all_timesteps[i]
            ep_returns = row_vals_curves[i]
            interpolated_returns = interpolate_returns(ep_returns, sc_timesteps, all_unique_timesteps)
            interped_row_vals_curves.append(interpolated_returns)
        
        row_vals_curves = interped_row_vals_curves

        # Now, each element in row_vals_curves is a Series of interpolated returns
        metric_curves_df = pd.DataFrame({i: vals for i, vals in enumerate(row_vals_curves)}).T
        metric_curves_df.columns = all_unique_timesteps
        metric_curves_df.index = row_index
        group_row_indices = [col for col in metric_curves_df.index.names if col != seed_name]
        metric_curves_mean = metric_curves_df.groupby(group_row_indices).mean()
        levels_to_drop = \
            [level for level in metric_curves_mean.index.names if
                metric_curves_mean.index.get_level_values(level).nunique() == 1]
        levels_to_keep = \
            [level for level in metric_curves_mean.index.names if
                metric_curves_mean.index.get_level_values(level).nunique() > 1]
        metric_curves_mean = metric_curves_mean.droplevel(levels_to_drop)

        # Create a line plot of the metric curves w.r.t. timesteps. Each row in the
        # column corresponds to a different line
        fig, ax = plt.subplots(
            # figsize=(20, 10)
        )
        for i, row in metric_curves_df.iterrows():
            ax.plot(row, label=str(i))
        ax.set_xlabel('Steps')
        ax.set_ylabel(key_name)
        # ax.legend()
        plt.savefig(os.path.join(CROSS_EVAL_DIR, name, f"{key_name.replace('/','_')}.png"))

        fig, ax = plt.subplots()
        # cut off the first and last 100 timesteps to remove outliers
        metric_curves_mean = metric_curves_mean.drop(columns=metric_curves_mean.columns[:25])
        metric_curves_mean = metric_curves_mean.drop(columns=metric_curves_mean.columns[-25:])
        columns = copy.deepcopy(metric_curves_mean.columns)
        # columns = columns[100:-100]
        for i, row in metric_curves_mean.iterrows():

            if len(row) == 0:
                continue

            # Apply a convolution to smooth the curve
            # row = np.convolve(row, np.ones(10), 'same') / 10
            # row = row[100:-100]
            # row = np.convolve(row, np.ones(10), 'same') / 10
            # turn it back into a pandas series
            row = pd.Series(row, index=columns)
            
            # drop the first 100 timesteps to remove outliers caused by conv
            # if row.index.shape[0] > 100:
            #     row = row.drop(row.index[:25])
            #     row = row.drop(row.index[-25:])
            

            ax.plot(row, label=str(i))
        metric_curves_mean.columns = columns
        ax.set_xlabel('Steps')
        ax.set_ylabel(key_name)

        # To get the ymin, drop the first timesteps where there tend to be outliers
        if metric_curves_mean.shape[1] > 100:
            ymin = metric_curves_mean.drop(columns=metric_curves_mean.columns[:100]).min().min()
        else:
            ymin = metric_curves_mean.drop(columns=metric_curves_mean.columns).min().min()

        # Can manually set these bounds to tweak the visualization
        # ax.set_ylim(ymin, 1.1 * np.nanmax(metric_curves_mean))
        # ax.set_ylim(bottom=-10, top=2)
        # ax.set_xlim(right=200_000_000)

        legend_title = ', '.join(levels_to_keep).replace('_', ' ')
        ax.legend(title=legend_title)
        fig_path = os.path.join(CROSS_EVAL_DIR, name, f"{key_name}_metric_curves_mean.png")
        os.makedirs(Path(fig_path).parent, exist_ok=True)
        plt.savefig(fig_path)
        print(f"Saved plot to {os.path.join(CROSS_EVAL_DIR, name, fig_path)}")


    # Save the dataframe to a csv
    # os.makedirs(os.path.join(CROSS_EVAL_DIR, name), exist_ok=True)
    # misc_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "misc_stats.csv")) 

    # Save to markdown
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.md"), 'w') as f:
    #     f.write(misc_stats_df.to_markdown())

    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.tex"), 'w') as f:
    #     f.write(misc_stats_df.to_latex())

    # seed_name = f'{algo}_seed'
    # Take averages of stats across seeds, keeping the original row indices
    # group_row_indices = [col for col in misc_stats_df.index.names if col != seed_name]
    # misc_stats_mean_df = misc_stats_df.groupby(group_row_indices).mean()

    # Save the dataframe to a csv
    # misc_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "misc_stats_mean.csv"))
    
    # Save to markdown
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.md"), 'w') as f:
    #     f.write(misc_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.tex"), 'w') as f:
    #     f.write(misc_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    # levels_to_drop = \
    #     [level for level in misc_stats_mean_df.index.names if 
    #      misc_stats_mean_df.index.get_level_values(level).nunique() == 1]
    # levels_to_keep = \
    #     [level for level in misc_stats_mean_df.index.names if
    #         misc_stats_mean_df.index.get_level_values(level).nunique() > 1]
    
    # Drop these rows
    # misc_stats_concise_df = misc_stats_mean_df.droplevel(levels_to_drop)

    # Save the dataframe to a csv
    # misc_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "misc_stats_concise.csv"))

    # Save to markdown
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.md"), 'w') as f:
    #     f.write(misc_stats_concise_df.to_markdown())

    # misc_stats_concise_df = clean_df_strings(misc_stats_concise_df)

    # Bold the maximum value in each column
    # styled_misc_stats_concise_df = misc_stats_concise_df.apply(format_num)

    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.tex"), 'w') as f:
    #     f.write(styled_misc_stats_concise_df.to_latex())

    
def cross_eval_il(cfg: SweepConfig, sweep_cfgs: Iterable[SweepConfig], hypers):
    # cross_eval_basic(name='il', sweep_configs=sweep_cfgs, hypers=hypers)
    _sweep_cfgs = []
    for s_cfg in sweep_cfgs:
        init_config(s_cfg)
        init_il_config(s_cfg)
        _sweep_cfgs.append(s_cfg)
    sweep_cfgs = _sweep_cfgs
    init_cross_eval_config(cfg, sweep_cfgs)
    cross_eval_misc(name=cfg.name, sweep_configs=sweep_cfgs, hypers=hypers)
    cross_eval_basic(name=cfg.name, sweep_configs=sweep_cfgs, hypers=hypers, algo='il')


def cross_eval_rl(cfg: SweepConfig, sweep_cfgs: Iterable[SweepConfig], hypers):
    # cross_eval_basic(name='il', sweep_configs=sweep_cfgs, hypers=hypers)
    _sweep_cfgs = []
    for s_cfg in sweep_cfgs:
        init_config(s_cfg)
        latest_evo_gen = init_il_config(s_cfg)
        init_rl_config(s_cfg, latest_evo_gen)
        _sweep_cfgs.append(s_cfg)
    sweep_cfgs = _sweep_cfgs
    init_cross_eval_config(cfg, sweep_cfgs)
    cross_eval_basic(name=cfg.name, sweep_configs=sweep_cfgs, hypers=hypers, algo='rl')
    cross_eval_misc(name=cfg.name, sweep_configs=sweep_cfgs, hypers=hypers, algo='rl')

    
def init_cross_eval_config(cfg: SweepConfig, sweep_cfgs: Iterable[EvoConfig]):
    # Note that we call this after initializing per-experiment directory names
    # Here we are just renaming for the sake of cross-eval tables/plots
    cfg.name = f'{cfg.algo}_{cfg.name}'
    for s_cfg in sweep_cfgs:
        if s_cfg.obs_window == -1:
            s_cfg.obs_window = s_cfg.map_shape[0] * 2 - 1