
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
from nonlinear_lodegp_control.helpers import load_system, Data_Def, Time_Def
from nonlinear_lodegp_control.feedback_linearization import Simulation_Config, learn_system_nonlinearities, Controller
from nonlinear_lodegp_control.gp import Linearizing_Control_5, CompositeModel
from nonlinear_lodegp_control.plotter import surface_plot, plot_trajectory,  plot_single_states

torch.set_default_dtype(torch.float64)
device = 'cpu'


system_name = "inverted_pendulum"

model_config = {
    'device': device,
    # 'model_path': model_path,
    'load': False,
    'save': False,
}


t  = 5
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
    Simulation_Config(sim_time, [-np.pi/2 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

    Simulation_Config(sim_time, [np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [-np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

    Simulation_Config(sim_time, [3* np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [-3 * np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    
    Simulation_Config(sim_time, [np.pi , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [0 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
]


control_gp_kwargs = {
    #Linearizing_Control_2
    'consecutive_training':False,
    #Linearizing_Control_4
    'b' : 110,
    'a' : torch.tensor([[a0],[a1]], dtype=torch.float64),
    'v' : torch.tensor([v], dtype=torch.float64),
    'controller':controller_0, # controller_0 # None  
    'noise': noise,
}


alpha, beta, control_gp = learn_system_nonlinearities(
    system, 
    sim_configs, 
    optim_steps, 
    ControlGP_Class = Linearizing_Control_5, #Linearizing_Control_5,# CompositeModel
    controlGP_kwargs = control_gp_kwargs,
    plot=False, 
    model_config=model_config,
    )

v = 0



# -----------------------------------------------------------------
# Plot surface of alpha and beta
# -----------------------------------------------------------------



l = 100
val = 3* torch.pi / 4
x_min = [-val, -val]
x_max = [torch.pi, val ]

test_points1, test_points2 = torch.meshgrid(
            torch.linspace(x_min[0], x_max[0], l),
            torch.linspace(x_min[1], x_max[1], l),
            indexing='ij'
        )
test_points = torch.stack([test_points1.flatten(), test_points2.flatten()], dim=-1)


if isinstance(control_gp, CompositeModel):
    train_inputs = control_gp.inducing_points
else:
    train_inputs = control_gp.train_inputs[0]

beta_system = torch.zeros(train_inputs.shape[0])
alpha_system = torch.zeros(train_inputs.shape[0])


for i in range(train_inputs.shape[0]):
    alpha_system[i] = system.alpha(train_inputs[i].numpy())
    beta_system[i] = system.beta(train_inputs[i].numpy())

with gpytorch.settings.observation_nan_policy('mask'):
    with torch.no_grad():
        test_alpha = alpha(test_points).squeeze()
        test_beta = beta(test_points, 0).squeeze()

fig_alpha = surface_plot(
    test_points1.numpy(),
    test_points2.numpy(),
    test_alpha.detach().numpy().reshape(l, l),
    train_inputs[:,0].numpy(),
    train_inputs[:,1].numpy(),
    alpha_system.numpy(),
    [r'$x_1$', r'$x_2$', r'$\alpha$'],
    [r'$\hat{\alpha}(\boldmath{x})$', r'${\alpha(\boldmath{x})}$']
)

fig_beta = surface_plot(
    test_points1.numpy(),
    test_points2.numpy(),
    test_beta.detach().numpy().reshape(l, l),
    train_inputs[:,0].numpy(),
    train_inputs[:,1].numpy(),
    beta_system.numpy(),
    [r'$x_1$', r'$x_2$', r'$\beta$'],
    [r'$\hat{\beta}(\boldmath{x})$', r'${\beta(\boldmath{x})}$']
)

plt.show()



# -----------------------------------------------------------------
# TEST CONTROLLER
# -----------------------------------------------------------------

test_controller = [
    Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=alpha, beta=beta),
    Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=system.alpha, beta=system.beta),
    controller_0
]

 

sim_time_u = Time_Def(0, 15, step=0.01)
rng = np.random.default_rng()
# position = (np.pi - 2* rng.random() * np.pi)
position = (np.pi/2)

x_0 = np.array([position , 0 ,0])

y_ref_control = np.zeros((sim_time_u.count))
ts = sim_time_u.linspace()

with gpytorch.settings.observation_nan_policy('mask'):
    with torch.no_grad():

        control_data = []
        for j in range(len(test_controller)):
            u_control = np.zeros_like(ts)
            sol = solve_ivp(
                system.stateTransition_2, 
                [sim_time_u.start, sim_time_u.end], 
                x_0[0:system.state_dimension], 
                method='RK45', 
                t_eval=ts, args=(sim_time.step, test_controller[j], u_control, y_ref_control),
                max_step=0.01
            )
            x = sol.y.transpose()

            solution = []
            for i in range (x.shape[1]):
                solution.append(x[:,i])

            solution.append(u_control)
            control_y = np.stack(solution, -1)

            control_data.append(Data_Def(ts.numpy(), control_y, system.state_dimension, system.control_dimension, sim_time_u))

calc_time_mean1, calc_time_std1 = test_controller[1].get_performance()

calc_time_mean2, calc_time_std2 = test_controller[2].get_performance()

print(f'exact: {calc_time_mean1} +/- {calc_time_std1}')
print(f'linear: {calc_time_mean2} +/- {calc_time_std2}')

figure = plot_single_states(
    control_data,
    ['GP', "exact feedback", r'$u_0$'],
    header= [r'$x_1$', r'$x_2$', r'$u$'], 
    yLabel=['angle (rad)', 'angular velocity (rad/s) ', 'force (N) '],
    line_styles=['-', '--', '-.'],
)


trajectory_plot = plot_trajectory(control_data, {}, ax_labels=['angle (rad)', 'angular velocity (rad/s)'], labels = ['GP', "exact feedback", r'$u_0$'])


plt.show()
