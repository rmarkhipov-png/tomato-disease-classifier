import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from leafdoctor.prepare_data import split_dataset
from leafdoctor.config import load_config
cfg = load_config()
split_dataset(cfg.data_dir, cfg.splits_dir, seed=cfg.seed)
