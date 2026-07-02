# LiteAgent — Hardware Inventory

This document lists the hardware components and specifications for both the local PC Workstation and the Raspberry Pi 5 edge device.

## Workstation & Edge Device Hardware Specs

| Device | Component | Specification | Measurement Method | Verified (Yes/No + Method) |
| :--- | :--- | :--- | :--- | :--- |
| **PC Workstation** | CPU | Intel Core i5-14600K (14 Cores, 20 Threads) | Windows PowerShell (`Get-CimInstance Win32_Processor`) | Yes — System Query |
| **PC Workstation** | GPU | NVIDIA GeForce RTX 5070 (12 GB VRAM) | `nvidia-smi` | Yes — System Query |
| **PC Workstation** | System RAM | 32 GB (2 x 16 GB Team Group DDR4/DDR5 @ 4800 MHz) | Windows PowerShell (`Get-CimInstance Win32_PhysicalMemory`) | Yes — System Query |
| **PC Workstation** | Storage speed (D:) | Read: 2763.59 MB/s, Write: 2314.44 MB/s (PCIe NVMe SSD) | Local custom Python benchmark writing/reading 256MB | Yes — Active Benchmark |
| **Raspberry Pi 5** | CPU | 64-bit quad-core Cortex-A76 processor | User-provided specs | Yes — User Confirmed |
| **Raspberry Pi 5** | GPU | Broadcom VideoCore VII (integrated, CPU-only inference) | User-provided specs | Yes — User Confirmed |
| **Raspberry Pi 5** | System RAM | 8GB LPDDR4X SDRAM | User-provided specs | Yes — User Confirmed |
| **Raspberry Pi 5** | Storage speed | `TODO: UNVERIFIED` (depends on SD Card class or PCIe NVMe Hat) | `dd` write/read test or `hdparm` | **No** — `UNVERIFIED` |

## Storage Benchmark Method for Raspberry Pi 5 (To be run by user)

To measure the actual storage read/write speed on the Raspberry Pi 5 to back the cold cache tier, run the following commands on the Pi. 

> [!IMPORTANT]
> The benchmark results depend heavily on the storage medium used. Please note whether the Pi 5 is running off a standard **microSD card** (expected ~20–80 MB/s) or a **PCIe NVMe SSD HAT** (expected ~300–800 MB/s), as this impacts cache eviction/swap latencies. Specify the medium alongside your results.

### Write Speed
```bash
dd if=/dev/zero of=test_write.tmp bs=1M count=256 oflag=direct
```
*Expected output shows write speed in MB/s.*

### Read Speed
```bash
dd if=test_write.tmp of=/dev/null bs=1M count=256 iflag=direct
```
*Expected output shows read speed in MB/s.*

*Clean up afterwards:* `rm test_write.tmp`
