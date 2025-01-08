# SimSDP - Generic Imaging Pipeline

SimSDP is an HPC resource allocation framework and simulation tool designed to optimize intra-node and inter-node resource allocation for scientific data processing. It combines **PREESM** for rapid prototyping and **SimGrid** for simulating inter-node communication and computation. 

The primary goal of SimSDP is to simulate the **Science Data Processor (SDP)** pipeline from the Square Kilometre Array (SKA), converting visibilities into output images on massive HPC architectures, handling datasets as large as the SKA requirements (xTB).

---

## Repository Structure

This repository contains six main directories, categorized by the number of frequencies processed in the pipeline:

### 1. Single-Frequency Projects
These projects simulate the SDP pipeline for processing data on **one frequency channel**.

- **`simsdp_g2g`**:  
  Simulates a generic **Graph-to-Graph (G2G)** transformation pipeline for single-frequency data.
  
- **`fft`**:  
  Focuses on the **Fast Fourier Transform (FFT)** step of the pipeline for single-frequency data.
  
- **`dft_imaging_pipeline`**:  
  Simulates the **Direct Fourier Transform (DFT)** imaging pipeline for single-frequency data.

### 2. Multi-Frequency Projects
These projects extend the SDP pipeline to process data on **multiple frequency channels**.

- **`simsdp_g2g_nfreq`**:  
  Simulates a generic **Graph-to-Graph (G2G)** transformation pipeline for multi-frequency data.
  
- **`fft_nfreq`**:  
  Focuses on the **Fast Fourier Transform (FFT)** step of the pipeline for multi-frequency data.
  
- **`dft_imaging_pipeline_nfreq`**:  
  Simulates the **Direct Fourier Transform (DFT)** imaging pipeline for multi-frequency data.

---

## Key Features

- **HPC Resource Allocation**:  
  Implements a model-based approach to optimize workload distribution between nodes using **PREESM**.
  
- **Simulation of Large Datasets**:  
  Handles simulations for datasets as large as those required by the SKA (xTB).
  
- **Modular Pipeline Design**:  
  Each pipeline step (G2G, FFT, DFT) is modular, supporting easy adaptation to different architectures and use cases.
  
- **Single and Multi-Frequency Support**:  
  Simulates pipelines for both single and multi-frequency imaging scenarios.

---

## Requirements

- **PREESM**: Used for rapid prototyping and workload partitioning.
- **SimGrid**: Provides a simulation environment for inter-node communication and computation.
- **Python/Java**: Development language support for the tools (adapt as needed).

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your_username>/SimSDP.git
   cd SimSDP

