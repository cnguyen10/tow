## Task weighting in meta-learning with trajectory optimisation
This repository contains the implementation of the paper *"Task weighting in meta-learning with trajectory optimisation"* published on Transactions on Machine Learning Research (August 2023).

### Requirements
The implementation relies on PyTorch. The code is tested on the following Python packages:
- torch 2.0.1
- torchvision
- torcheval
- higher
- aim (for experiment management and result visualisation)

### Running
One can run the code following the arguments specified in ```run.sh```. The result can be visualised by running the following command in the terminal:

```
aim up --repo logs
```

and follows the prompt in the terminal to open the browser at the corresponding port.