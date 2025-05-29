import gpytorch
import torch
from typing import List
import math

from nonlinear_lodegp_control.lodegp import optimize_gp
from nonlinear_lodegp_control.kernels import FeedbackControl_Kernel
from nonlinear_lodegp_control.mean_modules import FeedbackControl_Mean
from nonlinear_lodegp_control.warping import exponential_Gaussian, squared_gaussian

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                  covar_module=None,
                  mean_module=None
                  ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood:gpytorch.likelihoods.Likelihood, num_tasks:int):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def estimate(self, x:torch.Tensor):
        return self.likelihood(self(x)).mean
    
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
    
    def estimate(self, x:torch.Tensor):
        return self.likelihood(self(x)).mean


def y_ref_from_alpha_beta(alpha:gpytorch.distributions.Distribution, beta:gpytorch.distributions.Distribution, u:torch.Tensor):
        mean_x =  alpha.mean + beta.mean * u
        covar_x = u.unsqueeze(0) * beta.covariance_matrix * u.unsqueeze(1) + alpha.covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x + torch.eye(covar_x.shape[0]) * 1e-8)

class Linearizing_Control_5(gpytorch.models.ExactGP):
    def __init__(self, x:torch.Tensor, u:torch.Tensor, y_ref:torch.Tensor, b:float, a:torch.Tensor, v:torch.Tensor, variance, **kwargs):
        if isinstance(x, List):
            train_x = torch.cat(x, dim=0)
            train_u = torch.cat(u, dim=0)
            train_y = torch.cat(y_ref, dim=0) # TODOL output is dim 3: create two masked channels before
        else:
            train_x = x
            train_u = u
            train_y = y_ref
        train_y = torch.stack([torch.full_like(train_y, torch.nan), torch.full_like(train_y, torch.nan), train_y], dim=-1)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, )# noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
        # task_noise = torch.full((3 * var[0].shape[0] * len(var),), float('nan'))
        # task_noise[2::3] = torch.cat(var, dim=0).squeeze()
        # task_noise = torch.cat(var, dim=0).squeeze().repeat_interleave(3)

        # likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=3, noise=torch.tensor([1e-8,1e-8]), rank=3, has_task_noise=True, task_noise=task_noise)

        super().__init__(train_x,  train_y, likelihood)
        
        self.num_tasks = 3

        self.mean_module = FeedbackControl_Mean(b, a ,v)
        self.covar_module = FeedbackControl_Kernel(a, v)

        
        self.train_u = train_u

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def optimize(self, optim_steps:int, verbose:bool):
        training_loss, parameters =  optimize_gp(self, optim_steps, verbose)
        self.loss = training_loss[-1]
    
    def get_nonlinear_fcts(self):
        self.eval()
        self.likelihood.eval()
        def alpha(x):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                return self(torch.tensor(x)).mean[:, 0].unsqueeze(-1)

        def beta(x, u):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                return self(torch.tensor(x)).mean[:, 1].unsqueeze(-1)

        return alpha, beta
    
class VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
class CompositeModel(torch.nn.Module):
    def __init__(self, train_x:torch.Tensor, train_u:torch.Tensor, train_y:torch.Tensor, **kwargs):
        input_dim = 2
        if isinstance(train_x, List):
            self.train_u = torch.cat(train_u, dim=0)
            self.train_targets = torch.cat(train_y, dim=0)
            self.train_inputs = [torch.cat(train_x, dim=0)]
        else:
            self.train_u = train_u
            self.train_targets = train_y
            self.train_inputs = [train_x]

        inducing_length = math.floor(train_x[0].shape[0] / 2)
        train_z = torch.cat([x[0:inducing_length] for x in train_x], dim=0)
        #FIXME: How to choose inducing points from state space?
        # Choose inducing points by sampling uniformly from the input space
        # Assuming the input space is bounded, define the bounds
        x_min, x_max = train_x[:, 0].min(), train_x[:, 0].max()
        y_min, y_max = train_x[:, 1].min(), train_x[:, 1].max()

        # Generate a grid of inducing points within the bounds
        l = 5
        inducing_points_x, inducing_points_y = torch.meshgrid(
            torch.linspace(x_min, x_max, l),
            torch.linspace(y_min, y_max, l),
            indexing='ij'
        )
        inducing_points = torch.stack([inducing_points_x.flatten(), inducing_points_y.flatten()], dim=-1)
        self.inducing_points = inducing_points

        super().__init__()

        self._alpha = VariationalGP(inducing_points)
        self._log_beta = VariationalGP(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x, u):
        alpha = self._alpha(x)
        log_beta = self._log_beta(x)
        beta = squared_gaussian(log_beta)
        
        mean = alpha.mean + beta.mean * u
        covar = alpha.covariance_matrix + u.unsqueeze(0)*beta.covariance_matrix * u.unsqueeze(1)
        y_pred = gpytorch.distributions.MultivariateNormal(mean, covar)
        return y_pred, alpha, beta
    
    def optimize(self, optim_steps:int, verbose:bool):
        self._alpha.train()
        self._log_beta.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self._log_beta, num_data=self.train_inputs[0].size(0))

        train_loss = []

        for i in range(optim_steps * 10):
            optimizer.zero_grad()
            y_pred , alpha, log_beta = self(self.train_inputs[0], self.train_u)
            loss = -mll(y_pred, self.train_targets)
            train_loss.append(loss)
            loss.backward()
            if i % 50 == 0 and verbose:
                print(f"Iter {i}, Loss: {loss.item():.4f}")
            optimizer.step()

        self.loss = train_loss[-1].item()

    def get_nonlinear_fcts(self):
        self.eval()
        self.likelihood.eval()
        def alpha(x):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                return self._alpha(torch.tensor(x)).mean

        def beta(x, u):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                log_beta = self._log_beta(torch.tensor(x))
                _beta = squared_gaussian(log_beta)
            return _beta.mean
            # return torch.exp(log_beta.mean)
        
        return alpha, beta