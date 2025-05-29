import torch
import gpytorch

def exponential_Gaussian(gaussian: gpytorch.distributions.MultivariateNormal):
    # return gaussian
    mu = gaussian.mean
    Sigma = gaussian.covariance_matrix
    
    
    # exp_mu = torch.exp(mu + 0.5 * torch.diag(Sigma))  # shape: (d,)
    # outer_exp_mu = exp_mu.unsqueeze(1) * exp_mu.unsqueeze(0)  # shape: (d, d)
    # cov_Y = (torch.exp(Sigma) - 1) * outer_exp_mu  # shape: (d, d)
    # return gpytorch.distributions.MultivariateNormal(exp_mu, cov_Y)

    return gpytorch.distributions.MultivariateNormal(torch.exp(mu), gaussian._covar)

def squared_gaussian(gaussian: gpytorch.distributions.MultivariateNormal):
    # return gaussian
    alpha = 1
    mu = gaussian.mean
    Sigma = gaussian._covar # covariance_matrix # _covar

    mean = alpha + 1/2 * mu**2
    #covar = mu * Sigma.to_dense()  * mu
    covar = torch.diag(mu) @ Sigma @  torch.diag(mu)
    return gpytorch.distributions.MultivariateNormal(mean, covar) # + torch.eye(covar.shape[0]) * 1e-4