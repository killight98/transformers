from dataclasses import dataclass
from typing import Optional
import torch
import os


@dataclass
class ProfilerConfig:
    wait: int = 0
    warmup: int = 0
    active: int = 8
    repeat: int = 1
    skip_first: int = 0

    def get_schedule(self):
        return torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat,
                                       skip_first=self.skip_first)


class ProfilerWrapper:

    def __init__(self, device: str, config: Optional[ProfilerConfig]):
        self._config = config
        self._use_cuda = False
        _activities = [torch.profiler.ProfilerActivity.CPU]
        _extra_args = {}
        if "cuda" in device:
            _activities.append(torch.profiler.ProfilerActivity.CUDA)
            _extra_args['profile_memory'] = True
            self._use_cuda = True
        self._profiler = torch.profiler.profile(activities=_activities,
                                                schedule=self._config.get_schedule(),
                                                on_trace_ready=self._trace_handler, **_extra_args) if config else None

    def __enter__(self):
        if self._profiler:
            self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)

    def _trace_handler(self, prof):
        file_path = f"./torchshim_{os.uname()[1]}_{os.getpid()}_{prof.step_num}.json"
        prof.export_chrome_trace(file_path)

    def step(self):
        if self._profiler:
            self._profiler.step()
