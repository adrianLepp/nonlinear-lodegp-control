
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------
from nonlinear_lodegp_control.helpers import load_system, Data_Def, Time_Def
from nonlinear_lodegp_control.feedback_linearization import Simulation_Config, Controller, get_state_trajectories, get_linearizing_feedback
from nonlinear_lodegp_control.lodegp import LODEGP
from nonlinear_lodegp_control.plotter import plot_single_states

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "inverted_pendulum"

t  = 10
optim_steps = 100
downsample = 20
sim_time = Time_Def(0, t, step=0.01)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t, step=0.01)

a0 = 2
a1 = 3
v = 0

noise = torch.tensor([1e-3, 1e-3, 0], dtype=torch.float64)

system = load_system(system_name, a0=0, a1=0, v=1)
controller_0 = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]))

sim_configs = [
    Simulation_Config(sim_time, [np.pi/2  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-np.pi/2 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
]


model_config = {
    'device': device,
    # 'model_path': model_path,
    'load': False,
    'save': False,
}

control_gp_kwargs = {
    'consecutive_training':False,
    'b' : 110,
    'a' : torch.tensor([[a0],[a1]], dtype=torch.float64),
    'v' : torch.tensor([v], dtype=torch.float64),
    'controller':controller_0 # controller_0 # None  
}
data_names = [cfg.description for cfg in sim_configs]


# I
system_data = get_state_trajectories(system, sim_configs, control_gp_kwargs['controller'])

system_data_noise = [sys_data.add_noise(noise) for sys_data in system_data]


system_data_ = Data_Def(system_data[0].time, np.column_stack((system_data[0].y[:,0], system_data[0].y[:,1], np.gradient(system_data[0].y[:,1], 0.01))), state_dim=3, control_dim=0 )

# Calculate first and second order derivatives using finite differences
time = system_data_noise[0].time
dt = time[1] - time[0]



window_size = 5  # Define the size of the moving average window

sigma=1e-2
smoothed_signal = gaussian_filter1d(system_data_noise[0].y[:, 0], sigma=sigma)

first_derivative = savgol_filter(system_data_noise[0].y[:, 0], window_length=11, polyorder=3, deriv=1, delta=dt)
second_derivative = savgol_filter(system_data_noise[0].y[:, 0], window_length=11, polyorder=3, deriv=2, delta=dt)


derivative_data = Data_Def(
    time, 
    np.column_stack((smoothed_signal, first_derivative, second_derivative)), 
    system.state_dimension, 
    system.control_dimension, 
    sim_time
)



# II
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=system.dimension,
    # noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-4))
)
lodegp = LODEGP(None, None, likelihood, system.dimension, system.get_ODEmatrix())
lodegp_data = get_linearizing_feedback(lodegp, sim_configs, system_data_noise, optim_steps, model_config)

figure = plot_single_states(
    [derivative_data, lodegp_data[0], system_data_],
    ['smoothed', 'LODE-GP', 'true data', ],
    header= ['$\phi$', '$\dot{\phi}$', '$\ddot{\phi}$'], 
    yLabel=['angle (rad)', 'angular velocity (rad/s) ', 'angular acceleration ($rad/s^2$) '],
)

plt.show()