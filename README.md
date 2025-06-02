# nonlinear_lodegp_control

This project is based on the [LODE-GP](https://github.com/ABesginow/LODE-GPs) and was developed in my master thesis
*Physics-informed Gaussian Process Regression: Applications in Modelling and Control of Nonlinear Systems*.

For the part about Model Predictive Control (MPC), check the pulbication [Physics-informed Gaussian Processes for Model Predictive Control of Nonlinear Systems](https://arxiv.org/abs/2504.21377).
Everything else is not publicated yet.

Check the examples, on how to use the implemented methods.

## Requirements

For installation run 

```bash
git clone git@github.com:adrianLepp/nonlinear-lodegp-control.git
pip install -e ./nonlinear-lodegp-control
```


SageMath (https://www.sagemath.org/) needs to be installed seperately:

```bash
# Example install via conda (recommended):
conda install -c conda-forge
```

## Remark on gpytorch

The heteroskedastik noise for a multi-dimensional GP (as it is used for MPC), is not implemented in gpytorch yet.
See <https://github.com/cornellius-gp/gpytorch/issues/901> for proposed implementations. 
One possible implementation is present in this repository.

Calling the likelihood on another distribution than the training data is also not possible, due to a bug that is not fixed yet. 
See <https://github.com/cornellius-gp/gpytorch/issues/2630> for a description of this bug and a patch that can be applied to the source code.
However, it is normally not necessary to call the likelihood on something else than the training data.

