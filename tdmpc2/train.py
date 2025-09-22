import os
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

if torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='concept_discovery_multi_P_JEPA', config_path='.', version_base=None)
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	# assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	# ------------------------------------------------------------------
	# Protective check: abort if work directory already exists to avoid
	# accidental overwrites. No directory creation should have happened yet.
	# ------------------------------------------------------------------
	if os.path.exists(cfg.work_dir):
		eval_csv = cfg.work_dir / "eval.csv"
		if eval_csv.exists():
			print(colored('Work dir already contains eval.csv, indicating prior run:', 'red', attrs=['bold']))
			print(colored(str(eval_csv), 'yellow'))
			raise FileExistsError("Refusing to overwrite existing experiment directory.")
		# If eval.csv does not yet exist, treat directory as fresh (e.g., only .hydra present)

	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# Guard: original TD-MPC2 implementation must use single-layer model
	if getattr(cfg, 'original_tdmpc2_implementation', False) and int(getattr(cfg, 'num_jepa_layers', 1)) != 1:
		raise ValueError('original_tdmpc2_implementation=True requires num_jepa_layers == 1')

	# when the user tries to overwrite existing observations, ask for confirmation
	if cfg.save_obs_for_rankme:
		obs_file = f"/scratch//obs_data/{cfg.task}/{cfg.exp_name}/observations.pt"
		if os.path.exists(obs_file):
			overwrite = input(f"Observations file {obs_file} already exists. Overwrite? (y/n): ")
			if overwrite.lower() != 'y':
				print("Training cancelled to avoid overwriting existing observations.")
				exit()

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
