#!/usr/bin/env python
"""Download a MAD-Cars subsample: N cars x K evenly spaced views, resized on the fly.

usage: download_madcars.py [--cars 20000] [--views 10] [--min-views 4]
                           [--max-side 768] [--concurrency 48]
Rows are written to extra_data/madcars/meta/subsample.csv; images land in
extra_data/madcars/images/<car_id>/<view_id>.jpg. Re-running resumes safely.
"""
import argparse
import asyncio
import io
import time
from pathlib import Path

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "extra_data/madcars/meta/mad.csv"
SUBSAMPLE = ROOT / "extra_data/madcars/meta/subsample.csv"
OUT = ROOT / "extra_data/madcars/images"


def even_indices(n: int, target: int) -> list[int]:
    return sorted(np.unique(np.linspace(0, n - 1, min(target, n)).astype(int)).tolist())


def build_subsample(cars: int, views: int, min_views: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(META, usecols=["car_id", "view_id", "url", "brand", "model"])
    counts = df.groupby("car_id").size()
    eligible = counts[counts >= min_views].index
    rng = pd.Series(eligible).sample(n=min(cars, len(eligible)), random_state=seed).tolist()
    parts = []
    for car_id in rng:
        rows = df[df.car_id == car_id].sort_values("view_id")
        parts.append(rows.iloc[even_indices(len(rows), views)])
    sub = pd.concat(parts, ignore_index=True)
    sub.to_csv(SUBSAMPLE, index=False)
    return sub


async def fetch_one(session: aiohttp.ClientSession, url: str, path: Path, max_side: int) -> str:
    if path.exists():
        try:
            with Image.open(path) as img:
                img.verify()
            return "skip"
        except Exception:
            path.unlink()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return f"http_{resp.status}"
            content = await resp.read()
        with Image.open(io.BytesIO(content)) as img:
            img = img.convert("RGB")
            img.load()
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "wb") as f:
            await f.write(buf.getvalue())
        tmp.rename(path)
        return "ok"
    except Exception:
        if path.exists():
            path.unlink()
        return "error"


async def worker(session, queue, stats, max_side, t0, total):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        url, path = item
        result = await fetch_one(session, url, path, max_side)
        stats[result] = stats.get(result, 0) + 1
        done = sum(stats.values())
        if done % 1000 == 0:
            rate = done / (time.time() - t0)
            print(f"{done}/{total} ok={stats.get('ok', 0)} skip={stats.get('skip', 0)} err={stats.get('error', 0)} [{rate:.0f} img/s]", flush=True)
        queue.task_done()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cars", type=int, default=20000)
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--min-views", type=int, default=4)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    sub = build_subsample(args.cars, args.views, args.min_views, seed=42)
    if args.num_shards > 1:
        sub = sub[sub.car_id % args.num_shards == args.shard]
    print(f"subsample: {len(sub)} images, {sub.car_id.nunique()} cars (shard {args.shard}/{args.num_shards})", flush=True)

    queue = asyncio.Queue()
    stats = {}
    t0 = time.time()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker(session, queue, stats, args.max_side, t0, len(sub))) for _ in range(args.concurrency)]
        for row in sub.itertuples():
            path = OUT / str(row.car_id) / f"{row.view_id}.jpg"
            queue.put_nowait((row.url, path))
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
    print(f"DONE {stats} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
