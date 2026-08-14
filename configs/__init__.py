import importlib
from argparse import Namespace
from ast import literal_eval
from util.net import get_timepc


def _contains_key(container, key):
	if isinstance(container, dict):
		return key in container
	return hasattr(container, key)


def _get_key(container, key):
	if isinstance(container, dict):
		return container[key]
	return getattr(container, key)


def _set_key(container, key, value):
	if isinstance(container, dict):
		container[key] = value
	else:
		setattr(container, key, value)


def get_cfg(opt_terminal):
	opt_terminal.cfg_path = opt_terminal.cfg_path.split('.')[0].replace('/', '.')
	dataset_lib = importlib.import_module(opt_terminal.cfg_path)
	cfg = dataset_lib.cfg()
	# cfg = dataset_lib.cfg().__dict__
	# cfg_terms = {k: v for k, v in cfg_terms.items() if not k.startswith('_')}
	# ks = list(cfg_terms.keys())
	# for k in ks:
	# 	if k.startswith('_'):
	# 		del cfg_terms[k]
	# cfg = Namespace(**dataset_lib.__dict__)
	# cfg = Namespace(**cfg_terms)
	for key, val in opt_terminal.__dict__.items():
		if val is None:
			continue
		cfg.__setattr__(key, val)
	
	cssd_type_arg = ''
	if getattr(opt_terminal, 'cssd_type', None) is not None:
		cssd_type_arg = f' --cssd_type {opt_terminal.cssd_type}'
	cfg.command = f'python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c {cfg.cfg_path} -m {cfg.mode}{cssd_type_arg} --sleep {cfg.sleep} --memory {cfg.memory} --dist_url {cfg.dist_url} --logger_rank {cfg.logger_rank} {" ".join(cfg.opts)}'
	for opt in cfg.opts:
		cfg_ghost = cfg
		ks, v = opt.split('=', 1)
		ks = ks.split('.')
		try:
			v = literal_eval(v)
		except:
			v = v
		for i, k in enumerate(ks):
			if i == len(ks) - 1:
				_set_key(cfg_ghost, k, v)
			else:
				if not _contains_key(cfg_ghost, k):
					_set_key(cfg_ghost, k, Namespace())
				cfg_ghost = _get_key(cfg_ghost, k)
	cssd_type = getattr(opt_terminal, 'cssd_type', None)
	if cssd_type is not None:
		mamba_context_kwargs = dict(cfg.model.kwargs.get('mamba_context_kwargs', {}))
		mamba_context_kwargs['cssd_type'] = cssd_type
		cfg.model.kwargs['mamba_context_kwargs'] = mamba_context_kwargs
	cfg.task_start_time = get_timepc()
	return cfg


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg_path', default='configs/RD_test/rd_mvtec.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
	cfg_terminal = parser.parse_args()

	cfg = get_cfg(cfg_terminal)
	print(cfg)
