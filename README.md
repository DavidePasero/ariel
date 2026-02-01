![ariel-header](./docs/resources/ariel_header.svg)

# ARIEL: Autonomous Robots through Integrated Evolution and Learning

<!-- ## Requirements

* [vscode](https://code.visualstudio.com/)
  * [containers ext](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  * [container tools ext](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers)

* Container manager:
  * [podman desktop](https://podman.io/)
  * [docker desktop](https://www.docker.com/products/docker-desktop/)
  
* [vscode containers tut](https://code.visualstudio.com/docs/devcontainers/tutorial)

--- -->
## Installation and Running

This project uses [uv](https://docs.astral.sh/uv/).

First set **config.toml** with the settings you want to use in the experiments. With the command line you can only override genotype and task.

To run the experiments for reproducibility use:

```bash
cd ariel
uv venv
uv sync
python -m experiments.genomes.run_experiment \
--mode multiple \
--output path/to/genotype_task \
--num_runs n \
--genotype genotype \
--task task
```

To start a dash dashboard that will let you see the results of ONE EXPERIMENT (with multiple runs) use:
```bash
python -m experiments.genomes.multiple_runs_dashboard \
--db_paths \
path/to/genotype_task/run0.db \
path/to/genotype_task/run1.db \
path/to/genotype_task/run2.db \
path/to/genotype_task/run3.db \
path/to/genotype_task/run4.db \
path/to/genotype_task/run5.db \
path/to/genotype_task/run6.db \
path/to/genotype_task/run7.db \
path/to/genotype_task/run8.db \
path/to/genotype_task/run9.db \
```
Yes, you have to list all the .db files you want to include

To create the comparative plot between different groups of experiments (usually we compare the different genotypes on the same task) use:
```bash
python -m experiments.genomes.offline_comparative_plotter --experiment-dirs \
path/to/tree_task     \
path/to/lsystem_task  \
path/to/cppn_task     \
path/to/nde_task      \
--title "Fitness Comparison Between Four Genotypes on task" --names Tree L-System CPPN NDE"
```
In here you don't have to list all the individual .db files but just the parent folders in which all the runs of a certain experiments are stored. The code will automatically aggregate the results of all the runs.