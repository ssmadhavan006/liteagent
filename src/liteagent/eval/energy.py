"""
Energy measurement backends.

The project previously integrated `nvidia-smi` power draw unconditionally and
recorded the result as `energy_joules`. With inference running on the CPU that
sampled the card's ~10 W idle draw, producing a number that tracked wall-clock
latency and looked plausible in a plot while measuring nothing.

Two rules follow from that, and both are enforced here:

1. **Unavailable is not zero.** A backend that cannot measure returns `None`.
   `0.0` is a value that averages into results and silently biases them.
2. **A backend must refuse to report when it is not measuring the work.** The
   NVIDIA backend is unavailable unless layers are actually offloaded, because
   otherwise it is sampling an idle device.

Backend selection for this project:

    nvidia-smi   workstation only, and only with GPU offload enabled.
    rapl         Linux + Intel only. The Pi 5 is ARM (no RAPL) and Windows
                 needs a kernel driver, so it does not cover either target.
    external     wall-plug or inline USB-C meter exporting a timestamped CSV.
                 The only backend that works identically on both devices, so it
                 is the one that makes the edge/workstation comparison (H4)
                 meaningful.
    null         records `None`, used when nothing can measure honestly.

External meters typically sample at ~1 Hz, which is coarse for a task lasting a
few seconds. Prefer aggregating energy across a whole run and dividing by total
tokens over trusting any single short task's figure.
"""

import bisect
import csv
import os
import subprocess
import threading
import time


class EnergyMonitor:
    """Base backend. Reports `None` rather than a fabricated zero."""

    name = "null"

    def __init__(self):
        self.available = False

    def start(self) -> None:
        pass

    def stop(self) -> tuple[float | None, int]:
        """Returns (joules, sample_count). `None` joules means not measured."""
        return None, 0


class NvidiaSmiEnergyMonitor(EnergyMonitor):
    """
    Integrates GPU power draw by polling nvidia-smi.

    Only valid when layers are actually offloaded; otherwise it measures idle
    draw and must report nothing.
    """

    name = "nvidia_smi"

    def __init__(self, interval: float = 0.1, gpu_layers: int = 0):
        super().__init__()
        self.interval = interval
        self.power_samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread = None
        self.gpu_layers = gpu_layers
        self.available = bool(gpu_layers) and self._probe()

    def _probe(self) -> bool:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=0.5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _poll(self):
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=0.08,
                )
                if res.returncode == 0:
                    self.power_samples.append((start, float(res.stdout.strip())))
            except Exception:
                pass
            self._stop_event.wait(max(0.001, self.interval - (time.monotonic() - start)))

    def start(self):
        self.power_samples = []
        self._stop_event.clear()
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def stop(self) -> tuple[float | None, int]:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        if not self.available or not self.power_samples:
            return None, 0
        if len(self.power_samples) == 1:
            return self.power_samples[0][1] * self.interval, 1
        return _trapezoid(self.power_samples), len(self.power_samples)


class RaplEnergyMonitor(EnergyMonitor):
    """
    Reads Intel RAPL package energy counters from Linux powercap sysfs.

    Does not cover either target: the Pi 5 is ARM, and Windows exposes no
    sysfs. Provided so a Linux x86 host can be measured without an external
    meter, not as the project's primary method.
    """

    name = "rapl"
    _SYSFS = "/sys/class/powercap/intel-rapl:0/energy_uj"

    def __init__(self):
        super().__init__()
        self.available = os.path.exists(self._SYSFS) and os.access(self._SYSFS, os.R_OK)
        self._start_uj = None

    def _read_uj(self) -> int | None:
        try:
            with open(self._SYSFS, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None

    def start(self):
        self._start_uj = self._read_uj() if self.available else None

    def stop(self) -> tuple[float | None, int]:
        if not self.available or self._start_uj is None:
            return None, 0
        end = self._read_uj()
        if end is None:
            return None, 0
        delta = end - self._start_uj
        if delta < 0:
            # The counter wrapped; the true value is unknown, so report nothing.
            return None, 0
        return delta / 1_000_000.0, 2


class ExternalMeterEnergyMonitor(EnergyMonitor):
    """
    Integrates power from a wall-plug or inline USB-C meter's CSV export.

    The meter logs independently; this backend records the task window and
    integrates the samples that fall inside it. The CSV needs two columns,
    `timestamp` (unix seconds) and `watts`, and the meter's clock must be
    synchronised with this host.

    Because meters typically sample at ~1 Hz, a task shorter than a few seconds
    may contain one sample or none. `stop()` reports the sample count so a thin
    window can be discarded rather than silently trusted.
    """

    name = "external_meter"

    def __init__(self, csv_path: str, clock_offset_s: float = 0.0):
        super().__init__()
        self.csv_path = csv_path
        self.clock_offset_s = clock_offset_s
        self.available = bool(csv_path) and os.path.exists(csv_path)
        self._t0 = None
        self._samples: list[tuple[float, float]] = []
        if self.available:
            self._samples = self._load()

    def _load(self) -> list[tuple[float, float]]:
        rows = []
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append((float(row["timestamp"]) + self.clock_offset_s,
                                 float(row["watts"])))
        except Exception:
            return []
        rows.sort()
        return rows

    def reload(self) -> None:
        """Re-reads the CSV, for a meter still appending during the run."""
        if self.available:
            self._samples = self._load()

    def start(self):
        self._t0 = time.time()

    def stop(self) -> tuple[float | None, int]:
        if not self.available or self._t0 is None or not self._samples:
            return None, 0
        t1 = time.time()
        times = [t for t, _ in self._samples]
        lo = bisect.bisect_left(times, self._t0)
        hi = bisect.bisect_right(times, t1)
        window = self._samples[lo:hi]
        if len(window) < 2:
            # Too few samples inside the window to integrate honestly.
            return None, len(window)
        return _trapezoid(window), len(window)


def _trapezoid(samples: list[tuple[float, float]]) -> float:
    joules = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        joules += (samples[i][1] + samples[i - 1][1]) / 2.0 * dt
    return joules


def build_energy_monitor(
    source: str = "auto",
    gpu_layers: int = 0,
    meter_csv: str = None,
    interval: float = 0.1,
) -> EnergyMonitor:
    """
    Selects a backend.

    "auto" prefers an external meter, then GPU sampling when offload is on,
    then RAPL, and otherwise reports nothing rather than guessing.
    """
    if source == "none":
        return EnergyMonitor()
    if source == "external" or (source == "auto" and meter_csv):
        monitor = ExternalMeterEnergyMonitor(meter_csv or "")
        if monitor.available or source == "external":
            return monitor
    if source == "nvidia_smi" or source == "auto":
        monitor = NvidiaSmiEnergyMonitor(interval=interval, gpu_layers=gpu_layers)
        if monitor.available or source == "nvidia_smi":
            return monitor
    if source == "rapl" or source == "auto":
        monitor = RaplEnergyMonitor()
        if monitor.available or source == "rapl":
            return monitor
    return EnergyMonitor()
