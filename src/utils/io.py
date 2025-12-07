import os, json, yaml

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # simple "include" support
    if isinstance(data, dict) and "include" in data:
        base = load_yaml(data["include"])
        base.update({k:v for k,v in data.items() if k != "include"})
        return base
    return data

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def list_texts(dirpath: str):
    for root, _, files in os.walk(dirpath):
        for fn in sorted(files):
            if fn.endswith(".txt"):
                yield os.path.join(root, fn)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
