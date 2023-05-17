# Neurosurgeon Swarm (NS) Codebases

This document is a brief introduction to the three repos shipped from Neurosurgeon Swarm team: [NS Optimizer](https://github.com/huanchen-stack/obj-detection-optimizer/tree/yanmei), [NS Visualizers](https://github.com/Yanmeeei/NS-DOT-visualizers), and [tegra WATTS](https://github.com/huanchen-stack/tegraWATTS).

---

# NS Optimizer

NS Optimizer is an optimization toolset that, given model profiling data and other configurations (network bandwidth, memory constrains, etc.) as input, finds a set of optimized partitioning strategy for an NS model. There are several performance plotting tools in this repo as well. 

[link to the repo](https://github.com/huanchen-stack/obj-detection-optimizer/tree/yanmei)

---

# NS Visualizer
NS Visualizer is a visualization toolset for NS models and partitioning results. Those tools generate DOT codes that can be translated to graphs using any DOT visualizer (e.g. [Graphviz](https://dreampuf.github.io/GraphvizOnline/)). 

[link to the repo](https://github.com/Yanmeeei/NS-DOT-visualizers)

---

# tegraWATTS
tegraWATTS is a [tegrastats](https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html) parser that help provide layer by layer energy consumption in model inferences. 

Warning: The current tegraWATTS are tested only on single layer inferences. The code only takes care of jetson-nano and jetson-agx families.

[link to the repo](https://github.com/huanchen-stack/tegraWATTS)