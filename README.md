# SimSDP - Generic Imaging Pipeline

SimSDP is an HPC resource allocation framework and simulation tool designed to optimize intra-node and inter-node resource allocation for scientific data processing. It combines **PREESM** for rapid prototyping and **SimGrid** for simulating inter-node communication and computation. 

The primary goal of SimSDP is to simulate the **Science Data Processor (SDP)** pipeline from the Square Kilometre Array (SKA), converting visibilities into output images on massive HPC architectures, handling datasets as large as the SKA requirements (xTB).

In this project aims to use SimSDP to simulate the deployment of the Generic Imaging Pipeline with a single and multi-frequency scenarii on Multinode - Multicore architecture, on Multinode - MonoGPU architecture and (might be) on Multinode - MultiGPU architecture.


<img src="https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/raw/main/experimental_result_data/project_goal.png?ref_type=heads" style="zoom:100%;" />



---

## Repository Structure

This repository contains six main directories, categorized by the number of frequencies processed in the pipeline:

```plaintext
├── experimental_result_data/ # Spreadsheet diagrams of experimental results
├── simsdp-g2g-imaging-pipeline/
    ├── Algo/        # Dataflow Algorithms for pipeline steps 
    ├── Archi/       # HPC S-LAM Architecture models (e.g., HPC topology, node configurations)
    ├── Code/        # Source and generated code
    ├── Scenarios/   # Simulation scenarios 
    ├── Workflows/   # Workflow descriptions of the HPC resource allocation and simulation
├── simsdp-fft-imaging-pipeline/
├── simsdp-dft-imaging-pipeline/
├── simsdp-g2g-imaging-pipeline_nfreq/
├── simsdp-fft-imaging-pipeline_nfreq/
├── simsdp-dft-imaging-pipeline_nfreq/
```

> [!NOTE]
>
> The original Generic Imaging Pipeline has been formatted with the automatic cycle-unrolling integrated into PREESM to respect the dataflow paradigm in order to fit input requirement of the dataflow-based SimSDP framework.



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

- PREESM framework ([installation instructions](https://preesm.github.io/get/)) 
- Access to an HPC environment  

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@gitlab-research.centralesupelec.fr:dark-era/simsdp-generic-imaging-pipeline.git
   cd simsdp-generic-imaging-pipeline

2. Launch preesm and open folders from preesm: "File>open project from file system> browse `simsdp_g2g_imaging_pipeline` folder.

### Parameterized timing estimation
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    Timings definition consist in polynomials calculation, the procedure is the following.
    
    1. Generating instrumented code
    2. Running instrumented code varying the number of visibilities
    3. Executing the `plot_and_fit_averages.py` script
    
</details>

### Run SimSDP on a defined HPC architecture

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. Create a CSV file that will be exploit by the method to generate  multinode multicore HPC S-LAM files. Name this file “SimSDP_archi.csv”  and save it in the **Archi** folder. Your file should look like this:    

   ```bash
   Node name;Core ID;Core frequency;Intranode rate;Internode rate
   Node0;0;2000.0;472.0;9.42
   Node0;1;2000.0;472.0;9.42
   Node0;2;2000.0;472.0;9.42
   Node0;3;2000.0;472.0;9.42
   Node1;0;2000.0;472.0;9.42
   Node1;1;2000.0;472.0;9.42
   Node1;2;2000.0;472.0;9.42
   Node1;3;2000.0;472.0;9.42
   Node2;0;2000.0;472.0;9.42
   Node2;1;2000.0;472.0;9.42
   Node2;2;2000.0;472.0;9.42
   Node2;3;2000.0;472.0;9.42
   ```

2. Configure your network architecture: right click on your project “Preesm > generate custom architecture network”. Choose the network you  want. You can let it as default. It generates a XML file stored in the **Archi** folder (you can update it as you want). Your file should look like this:  
    ```html
   <!-- Cluster with shared backbone:3:4:1 -->
   <?xml version='1.0'?>
   
   <!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
   <platform version="4.1">
   <zone id="my zone" routing="Floyd">
   <cluster id="Dragonfly cluster" prefix="Node" radical="0-2" suffix="" speed="1f" bw="125MBps" lat="50us" topology="DRAGONFLY" topo_parameters="1,3;1,2;1,1;3">
         <prop id="wattage_per_state" value="90.0:90.0:150.0" />
         <prop id="wattage_range" value="100.0:200.0" />
     </cluster>
   </zone>
   </platform>
   ```
3. Run SimSDP: Right-click on the workflow “/Workflows/**hypervisor**.workflow” and select “Preesm > Run Workflow”, In the scenario selection wizard, select “/Scenarios/init_**.scenario

During its execution, the workflow will log information into the  Console of Preesm. When running a workflow, you should always check this console for warnings and errors (or any other useful information).

Additionnaly, the workflow execution generates intermediary dataflow graphs that can be found in the **/Algo/generated/** directory. The C code generated by the workflow is contained in the **/Code/generated/** directory. The simulated data are stored in the **/Simulation** directory.

4. A python notebook is provided in the SimSDP project to analyse the simulator generated files: Launch `jupyter notebook` and open “SimSDPproject/SimulationAnalysis.ipynb”. Make sure that the  CSVs are in the reading path. Load each code to display the trends with  your simulated data.
</details>
    
### Run SimSDP on a defined range of architectures
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    
1. Add a SimSDP_moldable.csv in the **Archi** folder  which should look something like this:

 | Parameters       | min        | max        | step       |
   | ---------------- | ---------- | ---------- | ---------- |
   | number of nodes  | 1          | 6          | 1          |
   | number of cores  | 1          | 6          | 1          |
   | core frequency   | 1          | 1          | 1          |
   | network topology | 1          | 4          | 1          |
   | node memory      | 1000000000 | 1000000000 | 1000000000 |

2. Run SimSDP: Right-click on the workflow “/Workflows/**hypervisor_dse**.workflow” and select “Preesm > Run Workflow”, In the scenario selection wizard, select “/Scenarios/init_**.scenario
3. Analyse the simulation with the`jupyter notebook` .
</details>

### Run the generated code
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
    
1. install the requirements:
```
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
2. Download [GLEAM](https://nasext-vaader.insa-rennes.fr/ietr-vaader/preesm/assets/sep_data.zip) (or whatever dataset).

3. copy past the data in Code/data/ folder. If it doesn't exist create a folder output/small/ inside.

4. Run the code and wait till your prompt display: `Process finished with exit code 0`.
(It could be long depending on the **NUM_MAJOR_CYCLE** and the **NUM_MINOR_CYCLE**).

#### Understanding the dataset
Open a terminal and launch `jupyter notebook`
and retrieve the file : /ska_sep_preesm/notebooks/dataset_prep.ipynb

Reload each code to update with your setup (you have to wait some time on several display, don't panic).

#### Visualizing the outut
At this stage verify that your output folder contains files such as :"cycle_0_clean_psf.csv"

```
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

## Experimental results

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
- Simulating generic imaging pipelines - 1 freq - CPU - balanced-workload based node partitioning - unoptimized code - [nVis = 3924480 , GRID_SIZE = 2048, nMinorCycle = 200] :

![](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/raw/main/experimental_result_data/1freq.png?ref_type=heads)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [**nVis** = 10xNUM_BASELINE:5xNUM_BASELINE:30xNUM_BASELINE , GRID_SIZE = 2048, nMinorCycle = 200]:

![](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/raw/main/experimental_result_data/simu_nvis.png?ref_type=heads)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [nVis = 3924480 , **GRID_SIZE** = 512:512:2560, nMinorCycle = 200]:

![](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/raw/main/experimental_result_data/simu_grid.png?ref_type=heads)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [nVis = 3924480 , GRID_SIZE = 2048, **nMinorCycle** = 50:50:250]:

![](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/raw/main/experimental_result_data/simu_minor.png?ref_type=heads)



- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [**nVis** = 1:784896:3924480 , nKernel = 108800, nMinorCycle = 200]:

[ToDo]

- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [nVis = 3924480 , **nKernel** = 1:21760:108800, nMinorCycle = 200]:

[ToDo]

- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [nVis = 3924480 , nKernel = 108800, **nMinorCycle** = 1:40:200]:

[ToDo]
</details>

## Citation

If you use this work, please cite the following publication:

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

## Contact  

For questions or feedback, please contact:  
- [Ophélie Renaud](mailto:ophelie.renaud@insa-rennes.fr)

