# multitask training of topographic RNN models

Pytorch implementation of multitask RNN training (original TensorFlow code [here](https://github.com/gyyang/multitask)):

> "Task representations in neural networks trained to perform many cognitive tasks." Guangyu Robert Yang, Madhura R. Joglekar, H. Francis Song, William T. Newsome & Xiao-Jing Wang (2019) [*Nature Neuroscience* Volume 22, pp. 297–306](https://www.nature.com/articles/s41593-018-0310-2)

This was forked from the [RNN_multitask](https://github.com/benhuh/RNN_multitask) repo, which was prepared for the [Harvard-MIT Theoretical and Computational Neuroscience Journal Club](https://compneurojc.github.io/).
A full RNN tutorial repo can be found [here](https://github.com/jennhu/rnn-tutorial).

Here, we modify the RNN to include spatial and sign-based constraints, following the work of Blauch, Behrmann, Plaut (2022). Additionally, we extend it to work with multi-area RNN models. 
> "A connectivity-constrained computational account of topographic organization in primate high-level visual cortex."  N.M. Blauch,M. Behrmann,& D.C. Plaut. (2022) Proc. Natl. Acad. Sci. U.S.A. 119 (3) e2112566119, https://doi.org/10.1073/pnas.2112566119

A demo notebook is provided at `multitask_demo.ipynb`, based off of the original repo.

Modified RNN code is found in `ei_rnn.py`. Wiring cost computations are performed within `wiring_cost.py`
