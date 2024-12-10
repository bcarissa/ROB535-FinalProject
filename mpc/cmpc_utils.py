import numpy as np

class BicycleSim:
    def __init__(self, param):
        # gen nominal trajectory
        self.Dim_state = 4
        self.Dim_ctrl  = 2
    
        self.L_f = param["L_r"]
        self.L_r = param["L_f"]
        self.h   = param["h"]
        self.T   = param["T"]

        self.a_lim = param["a_lim"]
        self.delta_lim = param["delta_lim"]
                
        self.Ns = int(np.ceil(self.T / self.h)) -1
    
    def Fun_dynamics_dt(self, x, u):

        xdot = np.zeros(4)        
        beta = np.arctan(self.L_r / (self.L_r + self.L_f) * np.arctan(u[1]))

        xdot[0] = x[3] * np.cos(x[2] + beta)
        xdot[1] = x[3] * np.sin(x[2] + beta)
        xdot[2] = x[3] / self.L_r * np.sin(beta)
        xdot[3] = u[0]
    
        xkp1 = x + xdot * self.h

        return xkp1
        
    def SimVehicle(self, x_bar, u_bar, preview, x0, controller):
    
        x_log = x_bar * 0.
        u_log = u_bar * 0.

        x_log[0, :] = x0

        x_bar_ext = np.concatenate((x_bar, np.ones((preview, x_log.shape[1])) * x_log[-1, :] ))
        u_bar_ext = np.concatenate((u_bar, np.ones((preview, u_log.shape[1])) * u_log[-1, :]))

        for k in range(self.Ns):
            
            u_act = controller(x_bar_ext[k:k+preview+1, :], u_bar_ext[k:k+preview, :], x_log[k, :])
            # print(k)
            x_log[k + 1, :] = np.squeeze(self.Fun_dynamics_dt(x_log[k, :],   u_act))
            u_log[k, :]     = np.squeeze(u_act)
            
            # print("x_log.shape", x_log.shape)
            # print("u_log.shape", u_log.shape)
            # print("u_bar.shape", u_bar.shape)
            # print("x_bar.shape", x_bar.shape)

        return x_log, u_log
    
    # def GenRef(self, alpha, beta):
    #     # generate a nominal trajectory
    #     x_bar = np.zeros((self.Ns + 1, self.Dim_state))
    #     u_bar = np.zeros((self.Ns    , self.Dim_ctrl))

    #     for k in range(self.Ns):
    #         u_act = np.array([ - 1 * (x_bar[k, 3] - 8 + 10 * np.sin(k / 20) + np.sin(k / np.sqrt(7)) ), 
    #                            np.cos(k / 10 / alpha) * 0.5 + 0.5 * np.sin(k / 10 / np.sqrt(beta))])

    #         u_act[0] = np.clip(u_act[0],  self.a_lim[0], self.a_lim[1])
    #         u_act[1] = np.clip(u_act[1],  self.delta_lim[0], self.delta_lim[1])
            
    #         u_bar[k, :]     = np.squeeze(u_act)
    #         x_bar[k + 1, :] = np.squeeze(self.Fun_dynamics_dt(x_bar[k, :],   u_act))
            
    #     print("u_bar.shape", u_bar.shape)
    #     print("x_bar.shape", x_bar.shape)
    #     print("self.ns",self.Ns)
    #     return u_bar, x_bar
    
    def GenRef(self):
        # generate a nominal trajectory
        # 加载路径文件
        x_bar = []
        with open("system/path.txt", "r") as f:
            for line in f:
                # 每行格式为: x, y, psi, v
                values = list(map(float, line.strip().split(",")))
                x_bar.append(values)
        x_bar = np.array(x_bar)  # 转换为 NumPy 数组

        # 加载控制输入文件
        u_bar = []
        with open("system/controls.txt", "r") as f:
            for line in f:
                # 每行格式为: a, delta
                values = list(map(float, line.strip().split(",")))
                u_bar.append(values)
        u_bar = np.array(u_bar)  # 转换为 NumPy 数组
        print("u_bar.shape",u_bar.shape)
        print("x_bar.shape", x_bar.shape)
        print("self.ns",self.Ns)
        return u_bar, x_bar