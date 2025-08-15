
import argparse, os, zipfile, shutil, pathlib
from bs4 import BeautifulSoup
from trafilatura import extract as trafi_extract
from pypdf import PdfReader
from markdown_it import MarkdownIt
from readability import Document
import lxml.html
from src.utils import normalize_ws

def save_txt(outdir, relpath, text):
    rel_txt = pathlib.Path(relpath).with_suffix('.txt')
    outpath = pathlib.Path(outdir) / rel_txt
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(text, encoding='utf-8')

def parse_pdf(fp):
    reader = PdfReader(fp)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def parse_html(raw):
    # try readability -> fall back to trafilatura
    try:
        doc = Document(raw)
        summary = doc.summary(html_partial=False)
        text = lxml.html.fromstring(summary).text_content()
        return text
    except Exception:
        t = trafi_extract(raw, include_comments=False) or ""
        return t

def parse_md(raw):
    md = MarkdownIt()
    tokens = md.parse(raw)
    text = []
    for t in tokens:
        if t.type == 'inline' and t.content:
            text.append(t.content)
    return "\n".join(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    with zipfile.ZipFile(args.zip) as z:
        z.extractall(args.out)

    # Normalize to .txt alongside originals for downstream pipeline
    for root, _, files in os.walk(args.out):
        for fn in files:
            fp = os.path.join(root, fn)
            ext = pathlib.Path(fn).suffix.lower()
            try:
                if ext in [".txt"]:
                    text = open(fp, encoding='utf-8', errors='ignore').read()
                    save_txt(args.out, os.path.relpath(fp, args.out), normalize_ws(text))
                elif ext in [".md", ".markdown"]:
                    raw = open(fp, encoding='utf-8', errors='ignore').read()
                    text = parse_md(raw)
                    save_txt(args.out, os.path.relpath(fp, args.out), normalize_ws(text))
                elif ext in [".html", ".htm"]:
                    raw = open(fp, encoding='utf-8', errors='ignore').read()
                    text = parse_html(raw)
                    save_txt(args.out, os.path.relpath(fp, args.out), normalize_ws(text))
                elif ext in [".pdf"]:
                    text = parse_pdf(fp)
                    save_txt(args.out, os.path.relpath(fp, args.out), normalize_ws(text))
                else:
                    # skip binaries, images, etc.
                    pass
            except Exception as e:
                print("Parse fail:", fp, e)

if __name__ == "__main__":
    main()
