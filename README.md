# ProGReST
## _Prototypical Graph Regression Soft Trees for Molecular Property Prediction_

In this work, we propose the novel Prototypical Graph Regression Self-explainable Trees (ProGReST) model, which combines prototype learning, soft decision trees, and Graph Neural Networks. In contrast to other works, our model can be used to address various challenging tasks, such as compound property prediction. In ProGReST, the rationale is obtained along with prediction due to the model's built-in interpretability.  Additionally, we introduce a new graph prototype projection process to accelerate model training. Finally, we evaluate PRoGReST on a wide range of chemical datasets for molecular property prediction and perform in-depth analysis with chemical experts to evaluate obtained interpretations. Our method achieves competitive results against state-of-the-art methods.


## How to use this code

### Instalation
Create new conda environment `pip install -r requirements.txt`


### R-MAT
If you want use R-MAT you have to install HuggingMolecules.
1. Download from `https://github.com/gmum/huggingmolecules/tree/add_rmat`
2. Update file  `huggingmolecules/src/setup.py`, change `torch==1.7.0` to `torch==1.10.0`
3. Install **Huggingmolecules** (Check paragraf `install huggingmolecules from the cloned directory`)

### Run Code
R-MAT and GCN file has main function with example how to setup args. 
To run code use:
- GCN `python gcn_reg.py`
- R-MAT `python rma_reg.py`
