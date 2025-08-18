"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
# todo: choose the coach
# from training.coach_aging_tune_psp import Coach
# from training.coach_aging_orig import Coach
# from training.coach_aging_tune_no_psp import Coach
from training.coach_aging_delta import Coach

import re

def main():
	opts = TrainOptions().parse()
	
    # Pick output directory.
	prev_run_dirs = []
	if os.path.isdir(opts.exp_dir):
		prev_run_dirs = [x for x in os.listdir(opts.exp_dir) if os.path.isdir(os.path.join(opts.exp_dir, x))]
	prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
	prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
	cur_run_id = max(prev_run_ids, default=-1) + 1
	opts.exp_dir = os.path.join(opts.exp_dir, f'{cur_run_id:05d}')
	assert not os.path.exists(opts.exp_dir)
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()

# import tempfile
# import torch
# import re

# def subprocess_fn(rank, opts, temp_dir):

#     # Init torch.distributed.
#     if opts.num_gpus > 1:
#         init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
#         if os.name == 'nt':
#             init_method = 'file:///' + init_file.replace('\\', '/')
#             torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
#         else:
#             init_method = f'file://{init_file}'
#             torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=opts.num_gpus)
            
#     # Execute training
#     coach = Coach(rank=rank, opts = opts)
#     coach.train()
            

# def launch_training(opts, desc, outdir):

#     # Pick output directory.
#     prev_run_dirs = []
#     if os.path.isdir(outdir):
#         prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
#     prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
#     prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
#     cur_run_id = max(prev_run_ids, default=-1) + 1
#     opts.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}') if desc else os.path.join(outdir, f'{cur_run_id:05d}')
#     assert not os.path.exists(opts.run_dir)

#     # Print options.
#     print()
#     print('Training options:')
#     print(json.dumps(vars(opts), indent=2))
#     print()

#     # Create output directory.
#     print('Creating output directory...')
#     os.makedirs(opts.run_dir)
#     with open(os.path.join(opts.run_dir, 'training_options.json'), 'wt') as f:
#         json.dump(vars(opts), f, indent=2, sort_keys=True)

#     opts.exp_dir = opts.run_dir

#     # Launch processes.
#     print('Launching processes...')
#     torch.multiprocessing.set_start_method('spawn', force=True)
#     with tempfile.TemporaryDirectory() as temp_dir:
#         if opts.num_gpus == 1:
#             subprocess_fn(rank=0, opts=opts, temp_dir=temp_dir)
#         else:
#             torch.multiprocessing.spawn(fn=subprocess_fn, args=(opts, temp_dir), nprocs=opts.num_gpus)
            
# def main():
#     opts = TrainOptions().parse()
#     os.makedirs(opts.exp_dir, exist_ok=True)
#     opts.num_gpus = opts.gpus
#     outdir = opts.exp_dir
#     if hasattr(opts, 'desc'):
#         desc = opts.desc
#     else:
#         desc = None
#     launch_training(opts, desc, outdir)


if __name__ == '__main__':
	main()
