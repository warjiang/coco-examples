from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote, unquote, urlparse, urlunparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from volcenginesdkarkruntime import Ark

load_dotenv()


DEFAULT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "your_collection_name")
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "your_qdrant_endpoint")
DEFAULT_EMBEDDING_MODEL = os.environ.get("ARK_EMBEDDING_MODEL", "multimodal_embedding_name")
DEFAULT_LIMIT = int(os.environ.get("QDRANT_LIMIT", "1"))
MAX_IMAGE_WIDTH_PX = int(os.environ.get("TERMINAL_IMAGE_MAX_WIDTH_PX", "256"))


qdrant_client = QdrantClient(url=DEFAULT_QDRANT_URL)
ark_client = Ark(api_key=os.environ.get("ARK_API_KEY"))


@dataclass
class Match:
    point_id: Any
    score: float
    payload: dict[str, Any]


def _clean_image_url(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    url = str(raw).strip()
    url = url.strip("\"'")
    url = url.replace("`", "").strip()
    if not url:
        return None
    return unquote(url)


def _download_to_temp(url: str) -> str:
    parsed = urlparse(url)
    encoded_path = quote(parsed.path, safe='/')
    fixed_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        encoded_path,
        parsed.params,
        parsed.query,
        parsed.fragment
    ))
    
    
    suffix = os.path.splitext(url.split("?", 1)[0])[1] or ".img"
    fd, path = tempfile.mkstemp(prefix="qdrant_match_", suffix=suffix)
    os.close(fd)
    print(f"downloading {fixed_url} to {path}")
    urllib.request.urlretrieve(fixed_url, path)
    return path


def _resize_with_sips(input_path: str, max_width_px: int) -> str:
    if shutil.which("sips") is None:
        return input_path
    fd, output_path = tempfile.mkstemp(prefix="qdrant_match_resized_", suffix=os.path.splitext(input_path)[1] or ".img")
    os.close(fd)
    try:
        subprocess.run(
            ["sips", "--resampleWidth", str(max_width_px), input_path, "--out", output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_path
    except Exception:
        try:
            os.remove(output_path)
        except Exception:
            pass
        return input_path


def _is_iterm2() -> bool:
    return os.environ.get("TERM_PROGRAM") == "iTerm.app"


def _display_image_iterm2_inline(image_path: str, max_width_px: int) -> bool:
    if not _is_iterm2():
        return False

    try:
        with open(image_path, "rb") as f:
            data = f.read()

        b64 = base64.b64encode(data).decode("ascii")
        name_b64 = base64.b64encode(os.path.basename(image_path).encode("utf-8")).decode("ascii")
        size = len(data)
        sys.stdout.write(
            f"\033]1337;File=name={name_b64};size={size};inline=1;width={max_width_px}px:{b64}\a\n"
        )
        sys.stdout.flush()
        return True
    except Exception:
        return False


def _display_image_in_terminal(image_path: str, max_width_px: int) -> bool:
    if _display_image_iterm2_inline(image_path, max_width_px):
        return True

    if shutil.which("imgcat") is not None:
        subprocess.run(["imgcat", "-W", f"{max_width_px}px", image_path], check=False)
        return True

    if shutil.which("wezterm") is not None:
        subprocess.run(["wezterm", "imgcat", "--width", f"{max_width_px}px", image_path], check=False)
        return True

    if shutil.which("kitten") is not None:
        subprocess.run(["kitten", "icat", "--transfer-mode", "stream", image_path], check=False)
        return True

    return False


def embed_text(text: str) -> list[float]:
    resp = ark_client.multimodal_embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=[
            {
                "type": "text",
                "text": text,
            }
        ],
    )
    return resp.data.embedding


def search_similar(embedding: list[float], limit: int) -> list[Match]:
    result = qdrant_client.query_points(
        collection_name=DEFAULT_COLLECTION_NAME,
        query=embedding,
        with_payload=True,
        limit=limit,
    ).points

    matches: list[Match] = []
    for p in result:
        payload = dict(p.payload or {})
        matches.append(Match(point_id=p.id, score=float(p.score or 0.0), payload=payload))
    return matches


def _print_match(m: Match) -> None:
    file_name = m.payload.get("file_name")
    image_url = _clean_image_url(m.payload.get("image_url")) or ""
    image_url = unquote(image_url)
    print(f"id={m.point_id} score={m.score:.6f} file_name={file_name!r}")
    if image_url:
        print(f"image_url={image_url}")


def main() -> None:
    print("Interactive Qdrant image search")
    print(f"QDRANT_URL={DEFAULT_QDRANT_URL}")
    print(f"COLLECTION={DEFAULT_COLLECTION_NAME}")
    print(f"MODEL={DEFAULT_EMBEDDING_MODEL}")
    print("Enter text to search (empty / 'exit' to quit).")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query or query.lower() in {"exit", "quit", ":q"}:
            return

        try:
            embedding = embed_text(query)
        except Exception as e:
            print(f"embedding failed: {e}")
            continue

        try:
            matches = search_similar(embedding, limit=DEFAULT_LIMIT)
        except Exception as e:
            print(f"qdrant query failed: {e}")
            continue

        if not matches:
            print("no matches")
            continue

        for idx, m in enumerate(matches, start=1):
            print(f"\n[{idx}/{len(matches)}]")
            _print_match(m)

            image_url = _clean_image_url(m.payload.get("image_url"))
            if not image_url:
                continue

            original_path: Optional[str] = None
            resized_path: Optional[str] = None
            try:
                original_path = _download_to_temp(image_url)
                resized_path = _resize_with_sips(original_path, MAX_IMAGE_WIDTH_PX)
                shown = _display_image_in_terminal(resized_path, MAX_IMAGE_WIDTH_PX)
                if not shown:
                    print(f"(no terminal image viewer found; saved to {resized_path})")
            except Exception as e:
                print(f"image display failed: {e}")
            finally:
                for p in {resized_path, original_path}:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass


if __name__ == "__main__":
    main()