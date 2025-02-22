# Automatic HPC simulation and algorithmic parameter browsing

As mention in the main `README`, this folder contain a set of radio-astronomy imaging dataflow pipelines.

- G2G
- DFT
- FFT

Dataflow programming consist in representing application with a graph where nodes, a.k.a actor, are computation and edges FIFO buffers. Synchronous Dataflow (SDF) consist in assigning an integer to each actor port, that correspond the rate of tokens produced and consumed by an actor. Parameterized Synchronous Dataflow integrate parameter to describe rate with expression that allow flexible representation of application.

In our radio-astronomy imaging pipeline, multiple parameter impact output quality and latency. To facilitate the algorithmic exploration we integrate moldable parameter in our pipelines.

![](/home/orenaud/Documents/CENTRAL SUPELEC REPO/simsdp-generic-imaging-pipeline/preesm_pipelines/top.svg)

The figure represent the G2G top dataflow graph, the blue triangles are the fixed parameters, the green triangles are the moldable parameter, containing a range of values to evaluate.
Running the compilation is described in the  the main `README`. This will browse all possible configuration simulating crucial metrics and stored each in log file in the `/Code/generated/` directory.
