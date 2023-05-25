# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/05/05
Description   :
"""
import yaml
from pathlib import Path
from argparse import Namespace

def read_yaml_file(fname):
	"""
	return namespace
	"""
	fname = ensure_path(fname)
	with fname.open('r', encoding='utf-8') as fid:
		contents = yaml.load(fid.read(), Loader=yaml.FullLoader)
	contents = Namespace(**contents)
	return contents


def write_yaml_file(fname, data):
	fname = ensure_path(fname)
	with fname.open("w", encoding="utf-8") as fid:
		yaml.dump(data, fid)


def ensure_path(path):
	"""Ensure string is converted to a Path.

	path (Any): Anything. If string, it's converted to Path.
	RETURNS: Path or original argument.
	"""
	if isinstance(path, str):
		return Path(path)
	else:
		return path


