import asyncio
import io
import runpy
import sys
import time
from pathlib import Path
from typing import cast

import aiohttp
import pandas as pd
from PIL import Image

from scripts import download_madcars


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, error_path: Path | None = None):
        self.content = content
        self.status = status
        self.error_path = error_path

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        if self.error_path is not None:
            self.error_path.write_bytes(b"partial")
            raise RuntimeError("interrupted download")
        return self.content


class FakeSession:
    def __init__(self, content: bytes, status: int = 200, error_path: Path | None = None):
        self.response = FakeResponse(content, status, error_path)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url, timeout=None):
        return self.response


def jpeg_bytes(size=(32, 20)):
    data = io.BytesIO()
    Image.new("RGB", size, (11, 22, 33)).save(data, format="JPEG")
    return data.getvalue()


def test_even_indices_and_subsample_selection(tmp_path, monkeypatch):
    assert download_madcars.even_indices(10, 3) == [0, 4, 9]
    assert download_madcars.even_indices(2, 10) == [0, 1]
    meta = pd.DataFrame(
        {
            "car_id": [10] * 4 + [11] * 4 + [12],
            "view_id": [1, 2, 3, 4] * 2 + [1],
            "url": ["https://example.org/car.jpg"] * 9,
            "brand": ["brand"] * 9,
            "model": ["model"] * 9,
        }
    )
    source = tmp_path / "mad.csv"
    meta.to_csv(source, index=False)
    monkeypatch.setattr(download_madcars, "META", source)
    subset = tmp_path / "meta/subsample.csv"
    sampled = download_madcars.build_subsample(1, 2, 4, 42, subset)
    assert sampled.car_id.nunique() == 1
    assert len(sampled) == 2
    assert subset.is_file()
    all_cars = download_madcars.build_subsample(0, 3, 4, 42, subset)
    assert all_cars.car_id.nunique() == 2
    assert len(all_cars) == 6


def test_fetch_one_skip_download_http_and_failure(tmp_path):
    content = jpeg_bytes()
    path = tmp_path / "car.jpg"
    session = cast(aiohttp.ClientSession, FakeSession(content))
    path.write_bytes(content)
    assert asyncio.run(download_madcars.fetch_one(session, "https://example.org", path, 16)) == "skip"
    path.write_bytes(b"corrupt")
    assert asyncio.run(download_madcars.fetch_one(session, "https://example.org", path, 16)) == "ok"
    with Image.open(path) as image:
        assert image.size == (16, 10)
    path.unlink()
    assert asyncio.run(download_madcars.fetch_one(session, "https://example.org", path, 64)) == "ok"
    with Image.open(path) as image:
        assert image.size == (32, 20)
    path.unlink()
    missing = cast(aiohttp.ClientSession, FakeSession(content, status=404))
    assert asyncio.run(download_madcars.fetch_one(missing, "https://example.org", path, 16)) == "http_404"
    assert not path.exists()
    broken = cast(aiohttp.ClientSession, FakeSession(b"not an image"))
    assert asyncio.run(download_madcars.fetch_one(broken, "https://example.org", path, 16)) == "error"
    interrupted = cast(aiohttp.ClientSession, FakeSession(content, error_path=path))
    assert asyncio.run(download_madcars.fetch_one(interrupted, "https://example.org", path, 16)) == "error"
    assert not path.exists()


def test_worker_counts_downloads_and_stops(tmp_path, capsys):
    async def work():
        queue = asyncio.Queue()
        path = tmp_path / "car.jpg"
        queue.put_nowait(("https://example.org", path))
        queue.put_nowait(None)
        stats = {"ok": 999}
        await download_madcars.worker(
            cast(aiohttp.ClientSession, FakeSession(jpeg_bytes())),
            queue,
            stats,
            16,
            time.time() - 1,
            1000,
        )
        await queue.join()
        return path, stats

    path, stats = asyncio.run(work())
    assert path.is_file()
    assert stats == {"ok": 1000}
    assert "1000/1000" in capsys.readouterr().out


def test_main_prepare_reuse_shard_and_module_guard(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    meta = tmp_path / "mad.csv"
    pd.DataFrame(
        {
            "car_id": [10, 10, 11, 11],
            "view_id": [1, 2, 1, 2],
            "url": ["https://example.org/car.jpg"] * 4,
            "brand": ["brand"] * 4,
            "model": ["model"] * 4,
        }
    ).to_csv(meta, index=False)
    monkeypatch.setattr(download_madcars, "ROOT", repo)
    monkeypatch.setattr(download_madcars, "META", meta)
    monkeypatch.setattr(download_madcars.aiohttp, "TCPConnector", lambda **kwargs: object())
    monkeypatch.setattr(
        download_madcars.aiohttp,
        "ClientSession",
        lambda connector: FakeSession(jpeg_bytes()),
    )
    argv = [
        "download_madcars.py",
        "--cars", "0", "--views", "2", "--min-views", "1", "--max-side", "16",
        "--concurrency", "2", "--dataset", "trial", "--num-shards", "2", "--shard", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--prepare-only"])
    asyncio.run(download_madcars.main())
    subset = repo / "extra_data/trial/meta/subsample.csv"
    assert subset.is_file()
    assert not (repo / "extra_data/trial/images").exists()
    monkeypatch.setattr(sys, "argv", argv)
    asyncio.run(download_madcars.main())
    assert (repo / "extra_data/trial/images/11/1.jpg").is_file()
    assert not (repo / "extra_data/trial/images/10/1.jpg").exists()
    asyncio.run(download_madcars.main())
    assert "reusing existing" in capsys.readouterr().out
    run = download_madcars.asyncio.run
    monkeypatch.setattr(download_madcars.asyncio, "run", lambda coroutine: coroutine.close())
    runpy.run_module("scripts.download_madcars", run_name="__main__")
    monkeypatch.setattr(download_madcars.asyncio, "run", run)
