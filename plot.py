import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


# Function to load TensorBoard logs
def load_tensorboard_logs(log_dir, keys):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()  # Load events from file

    stepss, valss = [], []

    # Retrieve scalars logged under the specified tag
    for k in keys:
        scalar_events = event_acc.Scalars(k)
        
        # Extract step and value information
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        stepss.append(steps)
        valss.append(values)
    
    return stepss, valss

# Plotting the data using matplotlib
def plot_data(log_dir, keys, stepss, valuess):
    search_stats = {}
    for key, steps, values in zip(keys, stepss, valuess):
        if 'search' in key:
            # Assert all valuess are the same
            assert all([v == values[0] for v in values])
            # Save this value to a text file
            with open(os.path.join(log_dir, f"{key.replace('/', '_')}.txt"), 'w') as f:
                f.write(str(values[0]))
            # And print it
            print(f"{key}: {values[0]}")
            search_stats.update({key: values[0]})
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label=f'{key}')
        plt.xlabel('Step')
        plt.ylabel(f'{key}')
        plt.title(f'{key}')
        print(key)
        if key == 'rl/ep_return':
            plt.ylim(bottom=-1)
        plt.legend()
        plt.grid(True)
        key = key.replace('/', '_')
        fig_path = os.path.join(log_dir, f"{key}.png")
        plt.savefig(fig_path)
        print(f"Plot saved to {fig_path}")

        # Save as dataframe, with rows indexed by step
        df = pd.DataFrame(data=np.vstack((steps, values)).T, index=None, columns=['steps', key])
        # Save to disk
        df.to_csv(os.path.join(log_dir, f"{key}.csv"))

    # Save search stats to a json
    import json
    with open(os.path.join(log_dir, 'search_stats.json'), 'w') as f:
        json.dump(search_stats, f, indent=4)
    


