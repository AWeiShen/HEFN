# HEFN

pytorch implementation of [Hierarchical High-Point Energy Flow Network for Jet Tagging](https://arxiv.org/abs/2308.08300)

A. the \hefn are original code of 2308.08300, only tree-graph corresponding Hierarchical Energy FLow Network is supported



B. the \hefn_tensor is an updated version of our code. It allows for the construction of any graph-structure corresponding HEFNs. 
We only give a few of minimal HEFNs exmaples (hundreds of trainable parameters and naive structures).


However, we want to note that high-dimensional tensors may impose a burden on CUDA memory, which could be relaxed by
1. using kinematic information of subjets rather than consitituents of jet
2. using more effiecient methods to save and operate symmetric tensors

              
    

            
