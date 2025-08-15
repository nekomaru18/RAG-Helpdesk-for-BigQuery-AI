
import re, os, hashlib, pathlib

def normalize_ws(text: str) -> str:
    text = re.sub(r'[\r\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def stable_id(*parts) -> str:
    h = hashlib.sha256('||'.join([str(p) for p in parts]).encode('utf-8')).hexdigest()
    return h[:12]

def guess_title_from_path(path: str) -> str:
    name = pathlib.Path(path).stem
    name = re.sub(r'[_\-]+', ' ', name)
    return name.strip().title()
