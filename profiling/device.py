import time

import torch


def as_device(device) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def is_cuda(device) -> bool:
    return as_device(device).type == "cuda"


def synchronize(device) -> None:
    target = as_device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


def reset_peak(device) -> None:
    target = as_device(device)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)


def allocated_bytes(device) -> int:
    target = as_device(device)
    if target.type != "cuda":
        return 0
    return int(torch.cuda.memory_allocated(target))


def peak_bytes(device) -> int:
    target = as_device(device)
    if target.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(target))


def reserved_bytes(device) -> int:
    target = as_device(device)
    if target.type != "cuda":
        return 0
    return int(torch.cuda.memory_reserved(target))


def mem_info(device) -> tuple[int, int] | None:
    target = as_device(device)
    if target.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(target)
    return int(free), int(total)


def memory_snapshot(device) -> dict:
    info = mem_info(device)
    return {
        "allocated_bytes": allocated_bytes(device),
        "peak_bytes": peak_bytes(device),
        "reserved_bytes": reserved_bytes(device),
        "free_bytes": None if info is None else info[0],
        "total_bytes": None if info is None else info[1],
    }


def pick_device(*, min_free_frac: float = 0.95, prefer=(), exclude=()) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    blocked = {int(index) for index in exclude}
    preferred = [int(index) for index in prefer]
    free_enough: list[int] = []
    for index in range(torch.cuda.device_count()):
        if index in blocked:
            continue
        free, total = torch.cuda.mem_get_info(index)
        if free / max(int(total), 1) > min_free_frac:
            free_enough.append(index)
    for index in preferred:
        if index in free_enough:
            return torch.device(f"cuda:{index}")
    if free_enough:
        return torch.device(f"cuda:{free_enough[0]}")
    return torch.device("cpu")


def timed(fn, device) -> tuple[object, float]:
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0
