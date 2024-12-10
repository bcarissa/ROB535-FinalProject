import numpy as np
import cvxpy as cp



class MPC_Controller:
    def __init__(self,param,dt=0.1):
        self.param = param
        self.L_r = param["L_r"]
        self.L_f = param["L_f"]
        
        self.dt = dt

    def calc_Jacobian(self,x, u, param):

        L_f = param["L_f"]
        L_r = param["L_r"]
        dt   = param["h"]

        psi = x[2]
        v   = x[3]
        delta = u[1]
        a   = u[0]

        # Jacobian of the system dynamics
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))

        A[0, 0] = 1.0
        A[0, 2] = -dt*v*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))
        A[0, 3] = dt*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))

        A[1, 1] = 1.0
        A[1, 2] = dt*v*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))
        A[1, 3] = dt*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))

        A[2, 2] = 1.0
        A[2, 3] = (dt*np.arctan(delta))/(((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)**(1/2)*(L_f + L_r))
        
        A[3, 3] = 1.0

        B[0, 1] =  -(L_r*dt*v*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r))))/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)*(L_f + L_r))
        B[1, 1] =   (L_r*dt*v*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r))))/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)*(L_f + L_r))
        B[2, 1] =  (dt*v)/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)**(3/2)*(L_f + L_r))
        B[3, 0] = dt
        return [A, B]
 
    def CMPC_Controller(self, x_bar, u_bar, x0):
        len_state = x_bar.shape[0]
        len_ctrl  = u_bar.shape[0]
        dim_state = x_bar.shape[1]
        dim_ctrl  = u_bar.shape[1]
        
        n_u = len_ctrl * dim_ctrl
        n_x = len_state * dim_state
        n_var = n_u + n_x

        n_eq  = dim_state * len_ctrl # dynamics
        n_ieq = dim_ctrl * len_ctrl # input constraints

        param = self.param
        a_limit = param["a_lim"]
        delta_limit = param["delta_lim"]
        
        #############################################################################
        #                    TODO: Implement your code here                         #
        #############################################################################
        
        # define the parameters
        Q = np.eye(4)  * 1
        R = np.eye(2)  * 0.00001
        Pt = np.eye(4) * 2 
        
        # define the cost function
        P = np.zeros((n_var, n_var))
        for k in range(len_ctrl):
            P[k * dim_state:(k+1) * dim_state, k * dim_state:(k+1) * dim_state] = Q
            P[n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl] = R
        
        P[n_x - dim_state:n_x, n_x - dim_state:n_x] = Pt
        P = (P.T + P) / 2
        q = np.zeros((n_var, 1))
        
        # define the constraints
        A = np.zeros((n_eq, n_var))
        b = np.zeros(n_eq)
        
        G = np.zeros((n_ieq, n_var))
        ub = np.zeros(n_ieq)
        lb = np.zeros(n_ieq)
        
        u_lb = np.array([a_limit[0], delta_limit[0]])
        u_ub = np.array([a_limit[1], delta_limit[1]])
        
        for k in range(len_ctrl):
            Jac_A, Jac_B = self.calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
            A[k * dim_state:(k+1) * dim_state,      k * dim_state:(k+1) * dim_state]       = Jac_A # AB[0:dim_state, 0:dim_state]
            A[k * dim_state:(k+1) * dim_state,  (k+1) * dim_state:(k+2) * dim_state]       = -np.eye(dim_state)
            A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]  = Jac_B # AB[0:dim_state, dim_state:]
            
            G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]    = np.eye(dim_ctrl)
            ub[k * dim_ctrl:(k+1) * dim_ctrl] = u_ub - u_bar[k, :]
            lb[k * dim_ctrl:(k+1) * dim_ctrl] = u_lb - u_bar[k, :]

        # Define and solve the CVXPY problem.
        x = cp.Variable(n_var)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                        [
                        G @ x <= ub,
                        lb <= G @ x,
                        A @ x == b,
                        x[0:dim_state] == x0 - x_bar[0, :]
                        ])
        prob.solve(verbose=False, max_iter = 10000)
        u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
        
    #     delta_s = cp.Variable((dim_state, len_state))  # State variables over time
    #     delta_u = cp.Variable((dim_ctrl, len_ctrl))    # Control variables over time
    #     t = cp.Variable()
    #     M = cp.Variable((2, 2), symmetric=True)
        
    #     # define the parameters
    #     Q = np.array([
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1],
    #     ])
    #     R = np.eye(2) * 0.00001
    #     Pt = np.eye(4) * 2 # Terminal Cost
        
    # # Define the cost function
    #     cost = cp.quad_form(delta_s[:, -1] - x_bar[-1], Pt)  # Terminal cost
    #     for i in range(len_ctrl):
    #         cost += cp.quad_form(delta_s[:, i] - x_bar[i], Q)  # State deviation cost
    #         cost += cp.quad_form(delta_u[:, i] - u_bar[i], R)  # Control deviation cost
            
    #     # Define constraints
    #     constraints = [delta_s[:, 0] == x0]  # Initial condition

    #     for i in range(len_ctrl):
    #         # Get system dynamics matrices A_k, B_k (you need a function to calculate these)
    #         A_k, B_k = self.calc_Jacobian(x_bar[i], u_bar[i], param)

    #         # Dynamic constraint: x(k+1) = A_k * x(k) + B_k * u(k)
    #         constraints = [M >> 0]
    #         constraints += [(delta_s[:, i+1] - x_bar[i+1]) == A_k @ (delta_s[:, i] - x_bar[i]) + B_k @ (delta_u[:, i] - u_bar[i])]
    #         # Input constraints: acceleration and steering limits
    #         constraints.append(delta_u[0, i] >= a_limit[0])  # Lower bound on acceleration
    #         constraints.append(delta_u[0, i] <= a_limit[1])  # Upper bound on acceleration
    #         constraints.append(delta_u[1, i] >= delta_limit[0])  # Lower bound on steering angle
    #         constraints.append(delta_u[1, i] <= delta_limit[1])  # Upper bound on steering angle

    #     prob = cp.Problem(cp.Minimize(cost), constraints)
    #     prob.solve(verbose=False, max_iter=10000)
    #     u_act = delta_u[:, 0].value
        #############################################################################
        #                            END OF YOUR CODE                               #
        #############################################################################
        return u_act
    
    def apply_control(self, u_act, state_):
    # 假设 u_act = [加速度, 方向盘角度]
        a, delta = u_act[0], u_act[1]
        beta = np.arctan(self.L_r / (self.L_r + self.L_f) * np.arctan(delta))

        # 使用动态方程更新机器人的状态
        # 根据当前状态 x 和控制输入 [a, delta] 更新状态 x_next
        x = state_
        psi = x[2]  # 当前朝向角度
        v = x[3]  # 当前速度

        # 更新位置和角度
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + v * np.cos(psi + beta) * self.dt
        x_next[1] = x[1] + v * np.sin(psi + beta) * self.dt
        x_next[2] = x[2] + (v / self.L_r) * np.sin(beta) * self.dt
        x_next[3] = x[3] + a * self.dt  # 速度更新

        return x_next