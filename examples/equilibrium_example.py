import gpytorch 
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_lodegp_control.kernels import *
import torch
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
from nonlinear_lodegp_control.lodegp import  optimize_gp, LODEGP
from nonlinear_lodegp_control.helpers import *
from nonlinear_lodegp_control.mean_modules import Equilibrium_Mean
from nonlinear_lodegp_control.plotter import plot_loss, plot_error, plot_states

torch.set_default_dtype(torch.float64)
device = 'cpu'


system_name = "nonlinear_watertank"
loss_file = '../data/losses/equilibrium.csv'



t  = 100
optim_steps = 300
downsample = 20 #  100 50 10
sim_time = Time_Def(0, t, step=0.1)
test_time = Time_Def(0, t-0, step=0.1)

noise = torch.tensor([1e-5, 1e-5, 1e-7])

u_e_rel = .2
u_rel = .3


system = load_system(system_name)

num_tasks = system.dimension

_ , x0 = system.get_ODEmatrix(u_e_rel)
system_matrix , equilibrium = system.get_ODEmatrix(u_e_rel)

x_0 = np.array(x0)

states = State_Description(equilibrium=torch.tensor(equilibrium), init=torch.tensor(x0))

u = np.ones((sim_time.count,1)) * u_rel * system.param.u

_train_x, _train_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u)
sim_data = Data_Def(_train_x, _train_y,system.state_dimension, system.control_dimension, sim_time)
train_data = sim_data.downsample(downsample).add_noise(noise)

# %% train


with gpytorch.settings.observation_nan_policy('mask'):


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks,
        has_global_noise=False, 
        noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )
    
    mean_module = Equilibrium_Mean(equilibrium, num_tasks)
    model = LODEGP(train_data.time, train_data.y, likelihood, num_tasks, system_matrix, mean_module)

    training_loss, _ = optimize_gp(model,optim_steps)

    # %% test

    test_x = test_time.linspace()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output = likelihood(model(test_x))
        lower, upper = output.confidence_region()
        
    _, _ = get_ode_from_spline(system, output.mean, test_x)

uncertainty = {
    'lower': lower.numpy(),
    'upper': upper.numpy()
}

x0_e = x0 - np.array(equilibrium)
ref_x, ref_y= simulate_system(system, x0_e[0:system.state_dimension], sim_time, u-equilibrium[-1], linear=True)
ref_data = Data_Def(ref_x.numpy(), ref_y.numpy() + np.array(equilibrium), system.state_dimension, system.control_dimension, sim_time)

_, _ = get_ode_from_spline(system, ref_data.y, ref_data.time)


test_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension, test_time, uncertainty)

error_gp = Data_Def(test_data.time, abs(test_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 
error_de = Data_Def(ref_data.time, abs(ref_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 

# Calculate RMSE for GP model
rmse_gp = np.sqrt(mean_squared_error(_train_y.numpy(), test_data.y))
std_gp = np.std(error_gp.y)

# Calculate RMSE for DE model
rmse_de = np.sqrt(mean_squared_error(_train_y.numpy(), ref_data.y))
std_de = np.std(error_de.y)

print(f"GP Model RMSE: {rmse_gp}, Standard Deviation: {std_gp}")
print(f"DE Model RMSE: {rmse_de}, Standard Deviation: {std_de}")

error_data_gp = error_gp.to_report_data()
error_data_de = error_de.to_report_data()

fig_loss = plot_loss({'loss':training_loss})

fig_error = plot_error(error_data_gp, error_data_de, ['x1', 'x2', 'u1'])
fig_results = plot_states([test_data, ref_data, train_data])

plt.show()