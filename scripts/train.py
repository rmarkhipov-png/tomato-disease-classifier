import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from leafdoctor.train import train
from leafdoctor.config import load_config
train(load_config())
