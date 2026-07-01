import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from leafdoctor.eval import evaluate
from leafdoctor.config import load_config
evaluate(load_config(), split="test")
