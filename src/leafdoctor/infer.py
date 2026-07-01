import io
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from leafdoctor.labels import ID2LABEL, NUM_CLASSES

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_model = None

def _load_model():
    global _model
    if _model is None:
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
        import pathlib
        model_path = pathlib.Path(__file__).parent.parent.parent / 'models' / 'model.pth'
        m.load_state_dict(torch.load(model_path, map_location='cpu'))
        m.eval()
        _model = m
    return _model

def predict(img_bytes: bytes, topk: int = 3):
    model = _load_model()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=topk)
    results = []
    for idx, prob in zip(top.indices.tolist(), top.values.tolist()):
        results.append({
            'label': ID2LABEL.get(idx, 'Неизвестно'),
            'confidence': round(prob * 100, 1),
            'idx': idx
        })
    return results

predict_image = predict
