# GPU Failure Analysis Platform (v1.0)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![HPC](https://img.shields.io/badge/HPC-ASPIRE2A-orange.svg)](https://www.nscc.sg/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains **Version 1.0** of the GPU Failure Analysis Platform, a high-performance computing (HPC) pipeline designed to simulate GPU telemetry, model failure dependencies, and detect hardware anomalies using machine learning.

This baseline version focuses on **binary failure classification** (Stable vs. Crash) using synthetic data generated via CUDA kernels on NVIDIA A100 GPUs.

## System Architecture

The platform consists of three core subsystems executed sequentially via PBS job scheduling:

1.  **CUDA Failure Simulation**
    Generates 1 million+ synthetic telemetry events (Voltage, Temperature, Memory) using parallel CUDA kernels. Injects deterministic failure logic (1% failure rate).

2.  **Graph-Based Root Cause Analysis**
    Models GPU subsystem dependencies (Power → Voltage → Thermal → Core) using NetworkX. Calculates centrality metrics to identify critical failure nodes.

3.  **Machine Learning Detection**
    -   **Supervised:** Random Forest Classifier for known failure patterns.
    -   **Unsupervised:** Isolation Forest for anomaly detection on unlabeled data.

## HPC Environment

This project was developed and executed on the **ASPIRE2A Supercomputer** (NSCC Singapore).

| Resource | Specification |
| :--- | :--- |
| **Cluster** | ASPIRE2A (Cray EX) |
| **GPU** | NVIDIA A100-SXM4-40GB |
| **CPU Allocation** | 16 vCPUs per GPU (Enforced Ratio) |
| **Memory** | 110 GB RAM per GPU |
| **Scheduler** | PBS Pro |
| **CUDA Version** | 12.4 (Driver) / 12.2 (Toolkit) |

## Methodology

### 1. Data Generation (CUDA)
Data is generated directly on the GPU using Numba CUDA kernels to maximize throughput. Each thread simulates an independent GPU execution scenario.

    # Example Kernel Logic
    if chance < 0.01:
        base_voltage += 0.5  # Spike
        base_temp += 30.0    # Overheat
        status_arr[idx] = 1  # Crash

### 2. Graph Modeling
A directed graph represents subsystem dependencies. Degree centrality identifies which subsystems are most influential in failure propagation.

    Nodes: [Power Delivery, Voltage Reg, Thermal Sensor, GPU Core, ...]
    Edges: [Power → Voltage, Voltage → Thermal, ...]

### 3. Machine Learning
Models are trained on the generated telemetry dataset (1M rows).

    - Training Set: 80%
    - Test Set: 20%
    - Features: Voltage, Temperature, Memory Utilization
    - Target: Failure Label (0=Stable, 1=Crash)

## Results (v1.0 Baseline)

The following results were obtained from the initial HPC job run (`pbs_output.txt`).

### GPU Simulation Stats
-   **Total Events:** 1,000,000
-   **Failure Rate:** ~1% (Binary)
-   **Generation Time:** < 2 minutes (GPU Accelerated)

### Graph Centrality Analysis
Top critical nodes identified in the failure propagation graph:

    - GPU Core: 0.8000
    - Voltage Reg: 0.6000
    - Thermal Sensor: 0.4000
    - Power Delivery: 0.2000
    - Memory Controller: 0.2000
    - Driver: 0.2000

### Machine Learning Performance
**Random Forest Classifier (Supervised)**

    precision    recall  f1-score   support
    0 (Stable)     1.00      1.00      1.00    198022
    1 (Crash)      1.00      1.00      1.00      1978
    accuracy       1.00                      200000
    macro avg      1.00      1.00      1.00    200000

**Isolation Forest (Unsupervised)**
-   **Anomalies Detected:** 10,000 events
-   **Contamination Rate:** 0.01

*Note: 100% accuracy is expected on synthetic data with deterministic rules. Real-world telemetry would yield lower accuracy, highlighting the importance of the Anomaly Detection module.*

## Project Structure

    gpu-failure-analysis-v1/
    ├── src/
    │   ├── cuda_simulator.py      # Component 1: Data Generation
    │   ├── graph_analyzer.py      # Component 2: Graph Analytics
    │   └── ml_detector.py         # Component 3: ML Pipeline
    ├── data/
    │   └── telemetry_1m.csv       # Generated Dataset
    ├── models/
    │   ├── rf_gpu_model.pkl       # Trained Random Forest
    │   └── iso_gpu_model.pkl      # Trained Isolation Forest
    ├── logs/
    │   ├── pbs_output.txt         # HPC Job Logs
    │   └── failure_graph.png      # Visualization
    └── run_job.pbs                # PBS Submission Script

## Usage

### Prerequisites
-   Access to ASPIRE2A HPC Cluster
-   Conda Environment (`test_ai_amd`)
-   Modules: `miniforge3`, `cuda/12.2.2`

### Running the Pipeline
Submit the job via PBS scheduler:

    qsub run_job.pbs

### Monitoring
Check job status:

    qstat -u $USER

View output logs:

    cat logs/pbs_output.txt

## Future Work (v2.0)
This repository represents the baseline (v1). Future iterations (v2) will include:
-   Multi-class classification (Power, Thermal, Memory failures).
-   XGBoost benchmarking.
-   Increased failure rate (5%) for richer analysis.
-   Real-time telemetry ingestion pipeline.

## License
This project is licensed under the MIT License.

## Acknowledgments
-   **NSCC Singapore** for providing ASPIRE2A HPC resources.
-   **NVIDIA** for CUDA toolkit and A100 GPU architecture.
