import csv
import os
import tempfile

import pytest

from liteagent.eval.energy import (
    EnergyMonitor,
    ExternalMeterEnergyMonitor,
    NvidiaSmiEnergyMonitor,
    build_energy_monitor,
)


def _write_meter_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "watts"])
        w.writerows(rows)


def test_null_monitor_reports_none_not_zero():
    """A fabricated 0.0 would average into results as a real measurement."""
    joules, samples = EnergyMonitor().stop()
    assert joules is None
    assert samples == 0


def test_nvidia_backend_unavailable_without_gpu_offload():
    """Sampling the GPU while inference runs on the CPU measures idle draw."""
    monitor = NvidiaSmiEnergyMonitor(gpu_layers=0)
    assert monitor.available is False
    monitor.start()
    joules, _ = monitor.stop()
    assert joules is None


def test_auto_selection_skips_gpu_when_offload_disabled():
    monitor = build_energy_monitor(source="auto", gpu_layers=0, meter_csv=None)
    assert monitor.name in ("null", "rapl", "external_meter")
    assert monitor.name != "nvidia_smi"


def test_external_meter_integrates_power_over_task_window():
    import time
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meter.csv")
        now = time.time()
        # Five samples at 1 Hz, constant 10 W: four 1 s intervals = 40 J.
        _write_meter_csv(path, [(now - 10 + i, 10.0) for i in range(5)])

        monitor = ExternalMeterEnergyMonitor(path)
        assert monitor.available
        monitor.start()
        # Back-date the window so it brackets the logged samples, standing in
        # for a task that actually ran while the meter was recording.
        monitor._t0 = now - 11
        joules, samples = monitor.stop()

        assert samples == 5
        assert joules == pytest.approx(40.0, rel=0.01)


def test_external_meter_refuses_to_integrate_a_thin_window():
    """At ~1 Hz a short task may contain too few samples to integrate."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "meter.csv")
        _write_meter_csv(path, [(1000.0, 10.0), (1001.0, 10.0)])
        monitor = ExternalMeterEnergyMonitor(path)
        monitor.start()
        joules, samples = monitor.stop()
        assert joules is None, "must not report energy from a window with <2 samples"
        assert samples < 2


def test_external_meter_missing_file_is_unavailable():
    monitor = ExternalMeterEnergyMonitor("does_not_exist.csv")
    assert monitor.available is False
    monitor.start()
    assert monitor.stop() == (None, 0)


def test_explicit_none_source_disables_measurement():
    monitor = build_energy_monitor(source="none", gpu_layers=-1)
    assert monitor.name == "null"
    assert monitor.stop()[0] is None
