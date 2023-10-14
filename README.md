# ACMP: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks
![ACMP](particle_interaction.png)
Our paper is at https://openreview.net/forum?id=4fZc_79Lrqs

## Introduction

 Neural message passing is a basic feature extraction unit for graph-structured data considering neighboring node features in network propagation from one layer to the next. We model such process by an interacting particle system with attractive and repulsive forces and the Allen-Cahn force arising in the modeling of phase transition. The dynamics of the system is a reaction-diffusion process which can separate particles without blowing up. This induces an Allen-Cahn message passing (ACMP) for graph neural networks where the numerical iteration for the particle system solution constitutes the message passing propagation. ACMP which has a simple implementation with a neural ODE solver can propel the network depth up to one hundred of layers with theoretically proven strictly positive lower bound of the Dirichlet energy. It thus provides a deep model of GNNs circumventing the common GNN problem of oversmoothing. GNNs with ACMP achieve state of the art performance for real-world node classification tasks on both homophilic and heterophilic datasets.


## Requirements

To install requirements:

```
pip install -r requirements.txt
```

or 

```
conda env create -f environment.yml
```

### Experiments
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora 
```

### Usage
The main message passing function is:

$$\frac{\partial}{\partial t}\mathbf{x}_i(t) = \alpha \odot \sum\limits _{j\in \mathcal{N}_i}(a(\mathbf{x}_i(t),\mathbf{x}_j(t))-\beta)(\mathbf{x}_j(t)-\mathbf{x}_i(t))+ \delta \odot \mathbf{x}_i(t)\odot(1-\mathbf{x}_i(t)\odot\mathbf{x}_i(t)).$$

Hyperparameter $\beta$ is a signal of the repulsive force, meaning that when $a_{ij} - \beta$ is negative, the two nodes repel one another. As shown in the following figure, $\beta$ exhibits completely different performances on datasets with two different levels of homophily. 

<div align="center">
    <img src="beta_study.png" alt="beta" width="50%">
</div>

There is a notebook in src folder, which show more experiment detail how we can control the particle system to overcome the oversommthing problem and heterophilic dataset problem.

## Comments 

- Our codebase for the graph diffusion models builds heavily on [Graph neural PDE](https://github.com/twitter-research/graph-neural-pde).
Thanks for open-sourcing!



## Citation 
If you consider our codes and datasets useful, please cite:
```
@inproceedings{
 wang2023acmp,
 title={{ACMP}: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks},
 author={Yuelin Wang and Kai Yi and Xinliang Liu and Yu Guang Wang and Shi Jin},
 booktitle={The Eleventh International Conference on Learning Representations },
 year={2023},
 url={https://openreview.net/forum?id=4fZc_79Lrqs}
}
```
