srun -n1 --tasks=1 --cpus-per-task=1 -t1:00:00 --gres=gpu:1 --mem=30000 --account=pr_174_tandon_advanced --pty /bin/bash