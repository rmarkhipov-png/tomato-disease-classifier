import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import uvicorn
uvicorn.run("leafdoctor.api:app", host="0.0.0.0", port=8000, reload=True)
