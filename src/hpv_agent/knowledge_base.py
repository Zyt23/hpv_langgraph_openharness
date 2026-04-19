from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from docx import Document
from pypdf import PdfReader

from .artifacts import ensure_dir, load_json, save_json
from .config import AppConfig


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_textlike(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_to_markdown(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".pdf":
        text = _read_pdf(src)
    elif src.suffix.lower() == ".docx":
        text = _read_docx(src)
    else:
        text = _read_textlike(src)
    dst.write_text(text, encoding="utf-8")


def build_knowledge_index(cfg: AppConfig) -> dict:
    run_dir = cfg.run_dir
    cache_dir = ensure_dir(run_dir / "knowledge_cache")
    index_path = run_dir / "knowledge_index.json"
    docs: List[dict] = []
    seen = set()
    paths: List[Path] = []
    for pattern in cfg.paths.knowledge_globs:
        import glob
        paths.extend(Path(p) for p in glob.glob(pattern))
    for path in paths:
        if path.is_dir():
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        if cfg.paths.data_root in path.resolve().parents:
            continue
        doc_id = path.stem
        md_path = cache_dir / f"{doc_id}.md"
        if not md_path.exists():
            normalize_to_markdown(path, md_path)
        docs.append({"doc_id": doc_id, "source_path": str(path.resolve()), "cache_md": str(md_path.resolve())})
    index = {"docs": docs}
    save_json(index_path, index)
    return index


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_doc_text(doc: dict) -> str:
    return Path(doc["cache_md"]).read_text(encoding="utf-8", errors="ignore")


def search_knowledge(index: dict, query: str, top_k: int = 3) -> List[dict]:
    q = set(tokenize(query))
    scored = []
    for doc in index["docs"]:
        text = load_doc_text(doc)
        toks = tokenize(text[:50000])
        overlap = len(q.intersection(set(toks)))
        if overlap == 0:
            continue
        scored.append((overlap, doc, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, doc, text in scored[:top_k]:
        out.append({"doc_id": doc["doc_id"], "snippet": text[:3000]})
    return out


def fetch_knowledge_context(index: dict, queries: List[str], doc_ids: List[str]) -> List[dict]:
    items: List[dict] = []
    for q in queries[:3]:
        items.extend(search_knowledge(index, q, top_k=2))
    doc_map = {d["doc_id"]: d for d in index["docs"]}
    for doc_id in doc_ids[:3]:
        if doc_id in doc_map:
            text = load_doc_text(doc_map[doc_id])
            items.append({"doc_id": doc_id, "snippet": text[:3000]})
    dedup = {}
    for item in items:
        dedup[item["doc_id"]] = item
    return list(dedup.values())
