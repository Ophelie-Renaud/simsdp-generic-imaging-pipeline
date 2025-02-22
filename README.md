# SimSDP - Generic Imaging Pipeline

SimSDP is an HPC resource allocation framework and simulation tool designed to optimize intra-node and inter-node resource allocation for scientific data processing. It combines **PREESM** for rapid prototyping and **SimGrid** for simulating inter-node communication and computation. 

The primary goal of SimSDP is to simulate the **Science Data Processor (SDP)** pipeline from the Square Kilometre Array (SKA), converting visibilities into output images on massive HPC architectures, handling datasets as large as the SKA requirements (179 Po :astonished:).

| 📝 **Note**                                                   |
| ------------------------------------------------------------ |
| Data = `N_ANT` x `N_CHANNEL` x `N_POL` x `BANDWIDTH` x `OBS_TIME` x `N_BYTE`<br /><br />(10min)=130000 × 4000 × 2 × 10⁶ × 600 × 4 = 2 Po<br />(12h)= 130000 × 4000 × 2 × 10⁶ × 43200 × 4 = 179 Po<br /><br /> **LOFAR (LOw Frequency ARray)** comparison:<br />(10min)=4992 × 256 × 2 × 195000 × 600 × 2 = 300 To<br />(12h)= 4992 × 256 × 2 × 195000 × 43200 × 2 = 21 Po<br /><br /> **NenuFAR (New Extension in Nançay Upgrading LOFAR)** comparison:<br />(10min)=1936 × 768 × 2 × 195000 × 600 × 2 = 348 To<br />(12h)= 1936 × 768 × 2 × 195000 × 43200 × 2 = 25 Po |


The aim of this project is to facilitate the comparison of algorithm describing the SDP using SimSDP.  As a dataflow-based tool, SimSDP takes into account the dataflow application graph as well as a csv file containing the execution times of the actors on each target to be simulated. Dataflow representation of  the **Generic Imaging Pipeline** with a single and multi-frequency scenarii are provided. Our goal is to simulate these pipelines on Multinode - Multicore architecture, on Multinode - MonoGPU architecture and (might be) on Multinode - MultiGPU architecture.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal.png" style="zoom:100%;" />

Algorithm performance is influenced by parameters such as the number of visibility points, grid size and number of minor cycles. We aim to identify and evaluate the influence of these parameters on scaling and assess whether a solution is viable in the context of SKA.

To achieve this, we provide tools that automatically estimate the execution time of dataflow actors for each targeted architecture and simulate algorithm performance by varying a set of parameter values automatically.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal2.png" style="zoom:30%;" />

---

## Repository Structure

This repository contains six main directories, categorized by the number of frequencies processed in the pipeline:

```plaintext
├── g2g_optim_actor/          # tools to build g2g lib
├── polynomial_timing/ 	      # CPU and GPU tool for timing calcultaion
├── preesm_pipelines/		  # g2g/dft/fft 1node/multinode architecture preesm projects
    ├── simsdp-g2g-imaging-pipeline/
        ├── Algo/        # Dataflow Algorithms for pipeline steps 
        ├── Archi/       # HPC S-LAM Architecture models (e.g., HPC topology, node configurations)
        ├── Code/        # Source and generated code
        ├── Scenarios/   # Simulation scenarios 
        ├── Workflows/   # Workflow descriptions of the HPC resource allocation and simulation
    ├── ...
├── param_code/ # resulting parameterized dataflow code
├── experimental_result_data/ # Spreadsheet diagrams of experimental results
```

## Key Features

- **HPC Resource Allocation**:  
  Implements a model-based approach to optimize workload distribution between nodes using **PREESM**.
  
- **Simulation of Large Datasets**:  
  Handles simulations for datasets as large as those required by the SKA (179 Po).
  
- **Modular Pipeline Design**:  
  Each pipeline step (G2G, FFT, DFT) is modular, supporting easy adaptation to different architectures and use cases.
  
- **Single and Multi-Frequency Support**:  
  Simulates pipelines for both single and multi-frequency imaging scenarios.

---

## Requirements

- **PREESM** framework ([installation instructions](https://preesm.github.io/get/)).

 | 📝 **Note**                                                   |
  | ------------------------------------------------------------ |
  | SimSDP may not be full integrated in the last PREESM release, to access it clone the [Preesm - github repository](https://github.com/preesm/preesm/) and branch `clustering2`. This is on going to be fixed. |

- Access to an HPC environment.

---

## Getting Started

Clone the repository:
```bash
git clone git@gitlab-research.centralesupelec.fr:dark-era/simsdp-generic-imaging-pipeline.git
cd simsdp-generic-imaging-pipeline
```
---

### Polynomial regression for static timing estimation :file_folder: `polynomial_timing`
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
.

This section describes a method for defining actor timings using a fitting function to facilitate algorithm comparison across varying parameters. The method involves building a set of samples (stored in :file_folder:`averages`) and computing a fitting function with polynomial regression for each actor.

The original method, developed by @sunrise.wang, is a manual approach that evaluates a limited number of data samples. Details of this method can be found on the :open_book: `wiki` page :page_facing_up: **Timing Modeling Manual Method**. Once the benchmark is set up, additional instructions are available in the :file_folder:`polynomials_timing` section under **SOTA**.

The proposed automated method, which extends :page_facing_up: [S. Wang, et al.](https://hal.science/hal-04361151/file/paper_dasip24_5_wang_updated-2.pdf) work, is documented in the :file_folder: ​`polynomials_timing` section under **Proposed Method**.

</details>

---

### Optimized G2G integration :file_folder: `g2g_optim_actor`
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

| 📝 **Note**                                                   |
| ------------------------------------------------------------ |
| The **ongoing work** consists in integrating the optimized G2G version into a dataflow actor. The current one is not optimized because ... 🙆‍♀️  <br /><br />The optimized version consists of C++ libraries from 📄 [N. Monnier, et al.](https://hal.science/hal-03725824/document).<br /><br />The steps include: <br />✅ Build the library <br />✅ Integrate it into the project <br />⬜ Translate the original Python code into C++ <br />⬜ Encapsulate it into a dataflow actor |

##### Including G2G libraries

1. generate the *.so librarie: `cd g2g_lib` > `cmake .` > `make`, the lib.so will be built in **build** :file_folder:.

2. create `libcpu_skytosky_single.h`:

```c
#ifndef LIBCPU_SKYTOSKY_SINGLE_H
#define LIBCPU_SKYTOSKY_SINGLE_H

#ifdef __cplusplus
extern "C" {
#endif

// Déclarations des fonctions exportées
void degridding_quad_pola(void);
void dgg_init_s2s(void);
void free_params(void);
void gridding_psf(void);
void gridding_quad_pola(void);
void init(void);
void s2s_quad_pola(void);
void s2s_single_pola(void);
void get_sky2sky_matrix_v0(struct interpolation_parameters* params);
void get_sky2sky_matrix_v1(struct interpolation_parameters* params);
void get_sky2sky_matrix_v3(struct interpolation_parameters* params);

#ifdef __cplusplus
}
#endif

#endif // LIBCPU_SKYTOSKY_SINGLE_H
```
3. include header: `#include "libcpu_skytosky_single.h"`.
4. compile with the lib: `gcc -o exe main.c -libcpu_skytosky_single.so -lcpu_skytosky_single -libcpu_skytosky_single.h`.

##### Optimized G2G dataflow actor implementation

 `cd g2g` > `cmake .` > `make`  >`./g2g`

</details>

---

### Run SimSDP on :file_folder: `preesm_pipelines`

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    .

SimSDP consists of three main steps:


- **Node-level partitioning**: Divides the dataflow graph into subgraphs, each assigned to an architecture node.
- **Thread-level partitioning**: Allocates resources for each subgraph on an architecture node.
- **Simulation**: Simulates both intra-node and inter-node behaviors.

SimSDP **has been [or is going to be] updated** to support multiple partitioning modes:

- <u>**Manual mode** (this project)</u>: Manually constructed subgraphs in the top-level graph are automatically mapped to architecture nodes.
- **Random mode**: The entire graph is partitioned among the available nodes and distributed with a random workload.
- **Balanced workload mode** (the original method, for more details see :open_book: `wiki`): The entire graph is partitioned among the available nodes and distributed with a balanced workload.

#### Simulating on <u>Multicore</u> & Multinode Architectures

##### Setup

To configure **manual mode**, follow these steps:

1. Open **PREESM** projects.
2. Navigate to `workflows/NodePartitioning.workflow`.
3. Select the `NodePartitioner` task.
4. Go to `Properties` > `Partitioning mode` :arrow_right: `manual`.

Dataflow pipelines are parameterized with moldable parameters. *(For details, see :page_facing_up: [A. Honorat, et al.](https://hal.science/hal-03752645/file/dasip22.pdf).)* This feature allows for parameter exploration, key metric simulations, and design space exploration (DSE). The moldable parameters include:

- `NUM_VIS` = {10× `NUM_BASELINES`; 15× `NUM_BASELINES`; 20× `NUM_BASELINES`; 25× `NUM_BASELINES`; 30× `NUM_BASELINES`}
  *where `NUM_BASELINES` = `NANT` × (`NANT` - 1) / 2 = 130816 (since `NANT` = 512)*
- `GRID_SIZE` = {512; 1024; 1536; 2048; 2560}
- `NUM_MINOR_CYCLE` = {50; 100; 150; 200; 250}

------

##### Single-Node Simulation

1. Launch **PREESM** and open the required project:

   - `File` > `Open Project from File System` > Browse to `simsdp_g2g_imaging_pipeline` folder.

2. Run the simulation:

   - Navigate to `Workflows/codegenMparmSelection.workflow`.
   - Right-click > `Preesm` > `Run Workflow`.
   - Select `1` or `6core.scenario`.

   :hourglass: During compilation, workflow logs will appear in the **Preesm Console**. Always check for warnings, errors, or any relevant information.

3. **Results** :bar_chart::

   - The generated **C code** and **moldable parameter logs** are located in the `/Code/generated/` directory.

------

##### Multi-Node Simulation

1. Launch **PREESM** and open the required project:

   - `File` > `Open Project from File System` > Browse to `simsdp_g2g_imaging_pipeline_nfreq` folder.

2. Run the simulation:

   - Navigate to `Workflows/codegen.workflow`.
   - Right-click > `Preesm` > `Run Workflow`.
   - Select `hypervisor.workflow`.

   :hourglass: During compilation, workflow logs will appear in the **Preesm Console**. Always check for warnings, errors, or any relevant information.

3. **Results** :bar_chart::

   - **Intermediate dataflow graphs**: Located in `/Algo/generated/`.
   - **Generated C code**: Found in `/Code/generated/`.
   - **Simulated data**: Stored in the `/Simulation/` directory.

4. **Further Analysis**:

   - A **Python notebook** is provided for analyzing simulation results.
   - Open `jupyter notebook` and load *SimSDPproject/SimulationAnalysis.ipynb*.
   - Ensure the required **CSV files** are accessible in the reading path.
   - Execute the notebook cells to visualize trends from the simulated data.

</details>

---

### Run the generated code from :file_folder: `param_code`
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    .

##### Basic execution

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    .

1. install the requirements:

```bash
sudo apt-get install libfftw3-dev

#BLAS
sudo apt-get install libblas-dev

#LAPACK
sudo apt-get install liblapack-dev
sudo apt-get install liblapacke-dev

#notebook to visualize
sudo apt-get install python3-pip
sudo apt install jupyter-notebook

#ASTROPY
sudo apt install python3-astropy
 
check: python3 -c "import astropy; print(astropy.__version__)"
```
2. Download [GLEAM](https://nasext-vaader.insa-rennes.fr/ietr-vaader/preesm/assets/sep_data.zip) (or whatever dataset) and copy past the data in :file_folder: `Code/data/`. If it doesn't exist create :file_folder: `output/small/` inside `data/`.

3. Run the code : `cmake .`  > `make` > `./sep` , and wait till your prompt display: `Process finished with exit code 0`(It could be long depending on the `NUM_MAJOR_CYCLE` and the `NUM_MINOR_CYCLE`).
    If you prefer CLion for as interface:
    - For the CPU version, run the CMakeList.txt, build :hammer: and Run  the code :arrow_forward:.
    
    - For GPU version, configure CMake:
    
      - install nvcc `sudo apt install nvidia-cuda-toolkit`, check the install `nvcc --version`.
    
      - Settings :gear:>Build, Execution, Deployment > CMake, add profile :heavy_plus_sign:, name `GIP_GPU`, CMake option `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (if you use the emulator: option `-DUSE_CUDA_EMULATOR=ON`, the emulator only allows you to check that the code is functional, execution will be slower than on a GPU).

</details>

---

##### Automating generated code execution varying parameter

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    .

| 📝 **Note**                                                   |
| ------------------------------------------------------------ |
| The **ongoing work** consists in scripting generated code execution varying parameter value (considering that the scheduling will not vary). Once the parameterized code is setting up the only command require is step 3. <br /><br />The steps include: <br />✅ Provide a script to compile generated code and store result (execution time) in log files<br />✅ Identify the generated code change in order to pass argument to facilitate parameter variation <br />⬜ Provide parametrized code for g2g,dft,fft pipeline (1 node) <br />⬜ Provide parametrized code for g2g,dft,fft pipeline (6 nodes) |

1. Copy past generated code from `preesm_pipeline` in folder `param_code`.
2. Apply the Following Modifications:
     - 🛠 **Edit** `preesm_gen.h`, modify the argument handling and define a struct for parameters:
        ```c
         /* if (arg != NULL) {
            printf("Warning: expecting NULL arguments\n");
            fflush (stdout);
          }*/
        
        typedef struct {
           int num_vis;
            int grid_size;
           int num_minor_cycle;
       } ThreadArgs;
       ```
     - 🛠 **Edit** `main.c`, pass parameters via command-line arguments and store them in a struct:
     	```c
         unsigned int launch(unsigned int core_id, pthread_t *thread, void* (*start_routine)(void*), void* arg) {
         ...
       pthread_create(thread, &attr, start_routine, arg);
       ...
       int main(int argc, char *argv[]) {
     	// Ensure correct number of arguments
     	if (argc != 4) {
     	printf("Usage: %s <NUM_VIS> <GRID_SIZE> <NUM_MINOR_CYCLE>\n", argv[0]);
     	return 1;
     	}
         // Parse command-line arguments
         int NUM_VIS = atoi(argv[1]);
         int GRID_SIZE = atoi(argv[2]);
         int NUM_MINOR_CYCLE = atoi(argv[3]);
       
         // Store them in a struct
         ThreadArgs args;
         args.num_vis = NUM_VIS;
         args.grid_size = GRID_SIZE;
         args.num_minor_cycle = NUM_MINOR_CYCLE;
         ...
         if (launch(CORE_ID[i], &coreThreads[i], coreThreadComputations[i],&args)) {
         ...
         coreThreadComputations[_PREESM_MAIN_THREAD_](&args);
       ```
     - 🛠 **Edit** `core0.c` etc..., update functions to use the parameter struct:
         ```c
         ThreadArgs* args = (ThreadArgs*) arg;  // Conversion du pointeur void* en ThreadArgs*
         int num_vis = args->num_vis;
         int grid_size = args->grid_size;
         int num_minor_cycle = args->num_minor_cycle;
     	
         // Replace all instances of hardcoded values:
         // int /*NUM_VIS*/ → num_vis
         // int /*GRID_SIZE*/ → grid_size
         // int /*NUM_MINOR_CYCLE*/ → num_minor_cycle
         ```
3. :arrow_forward: ​**Run** the experiment  script : 
     ```
     chmod +x run_experiments.sh
     ./run_experiments.sh g2g   # Run for g2g pipeline
     ./run_experiments.sh fft   # Run for fft pipeline
     ./run_experiments.sh dft   # Run for dft pipeline
     ```
      :boom: ​This will generate a log file with measured execution time in `g2g.csv` :arrow_right: copy the result file into: :file_folder: `experimental_result_data/moldable/measure/` .

</details>

---

##### Visualizing the outut

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    .

At this stage verify that your output folder contains files such as :"cycle_0_clean_psf.csv"

```bash
#Convert CSV files into fits files
python3 csvtoimage_all.py output/small/ fits/ ,

#install ds9
sudo apt install saods9

#Run
ds9 *.fits -lock frame wcs -zoom to fit
```
Filter: `/path to sep/ska_sep_preesm/Code/data/fits/*fits`

To reveal the contrasts:

- Color > Matplotlib > turbo (recommended by Sunrise)
- Color > Matplotlib > viridis / inferno (most popular in astro-papers)

![](https://github.com/Ophelie-Renaud/Imaging/blob/main/DS9_g2g_example1.png?raw=true)
</details>

</details>

---

## Experimental results :file_folder: `experimental_result_data`

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
.

How to retrieve the following results: `pip install scikit-learn` then `cd experimental_result_data/` > `python plot_simulation`.

| Pipeline        | Architecture            | SimSDP      |
| --------------- | ----------------------- | ----------- |
| G2G - ~~Clean~~ | 6 core CPU x86 - 1 node | semi-manual |

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/3D_comparison_g2g.png)

| 📝 **Analysis**                                               |
| ------------------------------------------------------------ |
| The **G2G** pipeline execution time scales with `GRID_SIZE` and `NUM_VIS`, following the complexity:  <br />Our method achieves a **RMSE of X**, improving over SOTA by better capturing real execution behavior. |

$$O(2n^2_g \log_2 n_g + 2n_v (n_{gk} + n_{dgk}))$$

Where:
-  $$n_g$$  is the number of grid points,
-  $$n_v$$  is the number of visibilities,
-  $$n_{gk}$$  is the size of the gridding kernel,
-  $$n_{dgk}$$ is the support size of the de-gridding kernel.




[ToDo]: feed with polynomial timings.

---

| Pipeline       | Architecture            | SimSDP      |
| -------------- | ----------------------- | ----------- |
| DFT- ~~Clean~~ | 6 core CPU x86 - 1 node | semi-manual |

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/simulation_dft.png)

| 📝 **Analysis**                                               |
| ------------------------------------------------------------ |
| The **DFT** pipeline execution time scales with `NUM_VIS`, following the complexity:  <br />Our method achieves a **RMSE of X**, improving over SOTA by better capturing real execution behavior. |

$$O(n^2_g \log_2 n_g + n_v  n_{gk})$$

Where:
-  $$n_g$$  is the number of grid points,
-  $$n_v$$  is the number of visibilities,
-  $$n_{gk}$$  is the size of the gridding kernel,
-  $$n_{dgk}$$ is the support size of the de-gridding kernel.

---

| Pipeline       | Architecture            | SimSDP      |
| -------------- | ----------------------- | ----------- |
| FFT- ~~Clean~~ | 6 core CPU x86 - 1 node | semi-manual |

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/simulation_fft.png)

| 📝 **Analysis**                                               |
| ------------------------------------------------------------ |
| The **FFT** pipeline execution time scales with `GRID_SIZE`, following the complexity:  <br />Our method achieves a **RMSE of X**, improving over SOTA by better capturing real execution behavior. |

$$O(n^2_g \log_2 n_g + n_v  n_{dgk})$$

Where:
-  $$n_g$$  is the number of grid points,
-  $$n_v$$  is the number of visibilities,
-  $$n_{gk}$$  is the size of the gridding kernel,
-  $$n_{dgk}$$ is the support size of the de-gridding kernel.

---

| Pipeline    | Architecture            | SimSDP      |
| ----------- | ----------------------- | ----------- |
| G2G - Clean | 6 core CPU x86 - 1 node | semi-manual |

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/3D_comparison_g2g_clean.png)

| 📝 **Analysis**                                               |
| ------------------------------------------------------------ |
| The **G2G** pipeline with **Clean** execution time scales with `GRID_SIZE` and  `NUM_MINOR_CYCLE` , following the complexity:  <br />Our method achieves a **RMSE of X**, improving over SOTA by better capturing real execution behavior. |

$$O(n^2_g + n_m  n_p)$$

Where:
-  $$n_g$$  is the number of grid points,
-  $$n_m$$  is the number of minor cycle,
-  $$n_p$$  is the support of the PSF,

</details>



---

## Citation

If you use this work, please cite the following publication:

```plaintext
# This publication for the Generic Imaging pipeline and the manual version of fitting function calculation
@InProceedings{10.1007/978-3-031-62874-0_5,
author="Wang, Sunrise
and Gac, Nicolas
and Miomandre, Hugo
and Nezan, Jean-Francois
and Desnos, Karol
and Orieux, Francois",
editor="Dias, Tiago
and Busia, Paola",
title="An Initial Framework for Prototyping Radio-Interferometric Imaging Pipelines",
booktitle="Design and Architectures for Signal and Image Processing",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="56--67"
}
```

```plaintext
# This publication for the design space exploration method
@inproceedings{renaud:hal-04608249,
  TITLE = {{Multicore and Network Topology Codesign for Pareto-Optimal Multinode Architecture}},
  AUTHOR = {Renaud, Oph{\'e}lie and Desnos, Karol and Raffin, Erwan and Nezan, Jean-Fran{\c c}ois},
  URL = {https://hal.science/hal-04608249},
  BOOKTITLE = {{EUSIPCO}},
  ADDRESS = {Lyon, France},
  ORGANIZATION = {{EURASIP}},
  PUBLISHER = {{IEEE}},
  PAGES = {701-705},
  YEAR = {2024},
  MONTH = Aug,
  DOI = {10.23919/EUSIPCO63174.2024.10715023},
  KEYWORDS = {HPC Simulation ; Dataflow model of computation ; Design space exploration DSE ; Network topology},
  PDF = {https://hal.science/hal-04608249v1/file/EUSIPCO_Multicore_and_Network_Topology_Codesign_for_Pareto_Optimal_Multinode_Architecture-2.pdf},
  HAL_ID = {hal-04608249},
  HAL_VERSION = {v1},
}
```

```plaintext
# Soon the SimSDP resource allocation method
```

```plaintext
# Soon the SimSDP proof of concept on HPC system with automated fitting function
```

## Wiki :open_book:

For references and so on take a look on the :open_book: `wiki`.

## Contact  

For questions or feedback, please contact:  
- [Ophélie Renaud](mailto:ophelie.renaud@ens-paris-saclay.fr)

## Acknowledgement

*This work was supported by DARK-ERA (ANR-20-CE46-0001-01).*
