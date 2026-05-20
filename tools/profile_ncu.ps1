<#
.SYNOPSIS
    Profile FlashNystrom kernel3 against the equivalent cuBLAS GEMMs with
    Nsight Compute, and print the metrics that explain any throughput gap.

.DESCRIPTION
    Runs tools/ncu_workload.py under ncu, capturing the two NVTX ranges
    (prof_fn = FlashNystrom forward, prof_cublas = cuBLAS GEMMs of the same
    shapes). Collects SpeedOfLight / Occupancy / Compute / Memory / WarpState
    sections and prints a filtered summary.

    PREREQUISITE: GPU performance counters must be readable. If you see
    ERR_NVGPUCTRPERM, run ONCE in an elevated PowerShell, then reboot:

        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" `
            -Name "RmProfilingAdminOnly" -Type DWord -Value 0

    After that this script does NOT need to be elevated.

.EXAMPLE
    ./tools/profile_ncu.ps1
    ./tools/profile_ncu.ps1 -B 1 -H 4 -N 16384 -D 64 -M 32
    ./tools/profile_ncu.ps1 -Python "C:\path\to\python.exe"
#>
param(
    [string]$Python = "C:\Users\athrv\miniconda3\envs\gpusamcts\python.exe",
    [int]$B = 1, [int]$H = 8, [int]$N = 4096, [int]$D = 128, [int]$M = 64,
    [string]$OutDir = "$env:TEMP\fn_ncu"
)
$ErrorActionPreference = "Stop"

# --- locate ncu.exe (Nsight Compute standalone, or bundled with the toolkit) ---
$ncu = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*\target\*\ncu.exe" `
        -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ncu) {
    $ncu = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\ncu.exe" `
            -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $ncu) { throw "ncu.exe not found. Install Nsight Compute (ships with the CUDA toolkit)." }
$ncu = $ncu.FullName
if (-not (Test-Path $Python)) { throw "Python not found: $Python  (pass -Python <path>)" }

$workload = Join-Path $PSScriptRoot "ncu_workload.py"
if (-not (Test-Path $workload)) { throw "workload not found: $workload" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$rep = Join-Path $OutDir "fn_vs_cublas"

Write-Host "ncu:      $ncu"
Write-Host "python:   $Python"
Write-Host "workload: $workload  (B=$B H=$H N=$N D=$D m=$M)"
Write-Host "out:      $rep.ncu-rep`n"

# Only profile the kernels we care about (FN kernel3 + cuBLAS GEMMs), within
# the NVTX ranges. The cuBLAS kernel name varies by version/arch, so the regex
# is deliberately broad. If the cuBLAS side comes up empty, widen it or drop
# --kernel-name entirely and re-run.
$kernelRegex = "regex:(kernel3_(partial|combine|fused)|gemm|cutlass|ampere|xmma|16816|elementwise)"
$sections = @("SpeedOfLight", "Occupancy", "ComputeWorkloadAnalysis",
              "MemoryWorkloadAnalysis", "WarpStateStats")
$secArgs = $sections | ForEach-Object { "--section"; $_ }

Write-Host "Profiling... ncu replays each kernel many times to read counters; this takes a few minutes."
& $ncu --target-processes all `
       --nvtx --nvtx-include "prof_fn/" --nvtx-include "prof_cublas/" `
       --kernel-name $kernelRegex `
       @secArgs `
       --force-overwrite --export $rep `
       $Python $workload $B $H $N $D $M
if ($LASTEXITCODE -ne 0) {
    throw "ncu failed (exit $LASTEXITCODE). If ERR_NVGPUCTRPERM, enable counters (see this script's header) and reboot."
}

Write-Host "`n==================== SUMMARY (paste this back) ===================="
& $ncu --import "$rep.ncu-rep" --page details |
    Select-String -Pattern @(
        "void flash_nystrom", "gemm", "cutlass", "ampere",     # kernel names
        "Duration", "Compute \(SM\)", "Memory Throughput",
        "Achieved Occupancy", "Tensor", "DRAM Throughput",
        "Stall"                                                 # warp stall reasons
    )
Write-Host "`nFull report: $rep.ncu-rep  (open in the Nsight Compute UI for the complete picture)"
