import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from leafdoctor.infer import load_model, predict
from leafdoctor.config import load_config
if len(sys.argv) < 2:
    print("Usage: python scripts/predict.py image.jpg")
    sys.exit(1)
cfg = load_config()
ckpt = pathlib.Path(f"artifacts/{cfg.project_name}/{cfg.version}/best.pt")
model, device = load_model(ckpt, cfg.model_name)
result = predict(sys.argv[1], model, device)
print(json.dumps(result, indent=2, ensure_ascii=False))
