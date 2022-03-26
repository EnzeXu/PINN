import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.cluster import KMeans
import random
from torch.backends import cudnn
from pyDOE import lhs
from torch import optim


class SimpleNetwork(nn.Module):
    def __init__(self, opt=None):
        super(SimpleNetwork, self).__init__()
        self.setup_seed(0)
        self.model_name = "SimpleNetwork"
        self.fc_start = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 32),
            'sig1': nn.Tanh(),
        }))

        self.fc_middle = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(64, 16),
            'sig1': nn.Tanh(),
        }))

        self.fc_end = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(8, 1),
            'sig1': nn.Tanh(),
        }))

        # self.fc_start = nn.Sequential(OrderedDict({
        #     'lin1': nn.Linear(1, 32),
        #     'sig1': nn.Tanh(),
        #     'lin2': nn.Linear(32, 64),
        #     'sig2': nn.Tanh(),
        # }))
        #
        # self.fc_middle = nn.Sequential(OrderedDict({
        #     'lin1': nn.Linear(128, 64),
        #     'sig1': nn.Tanh(),
        #     'lin2': nn.Linear(64, 128),
        #     'sig2': nn.Tanh(),
        # }))
        #
        # self.fc_end = nn.Sequential(OrderedDict({
        #     'lin1': nn.Linear(64, 32),
        #     'sig1': nn.Tanh(),
        #     'lin2': nn.Linear(32, 1),
        #     'sig2': nn.Tanh(),
        # }))

    def forward(self, inputs):
        u_old, v_old = torch.chunk(inputs, 2, 1)
        u_middle = self.fc_start(u_old)
        v_middle = self.fc_start(v_old)
        middle_input = torch.cat((u_middle, v_middle), 1)
        middle_output = self.fc_middle(middle_input)
        middle_output_chunk = torch.chunk(middle_output, 2, 1)
        u_end_input, v_end_input = middle_output_chunk[0], middle_output_chunk[1]
        u_new = self.fc_end(u_end_input)
        v_new = self.fc_end(v_end_input)
        outputs = torch.cat((u_new, v_new), 1)
        # print("u_middle:", u_middle.shape, "middle_input:", middle_input.shape, "middle_output:", middle_output.shape, "u_end_input:", u_end_input.shape, "u_new:", u_new.shape, "outputs:", outputs.shape)
        return outputs

    @staticmethod
    def loss(y_hat, y):
        return torch.mean(torch.square(y_hat - y))
        # return F.mse_loss(torch.cat((u_hat, v_hat), 1), torch.cat((u, v), 1))
        # return torch.abs(u_hat - u) + torch.abs(v_hat - v)  # F.mse_loss(x_hat, x) + beta * self.kl_div(rho)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True


class SimpleNetworkPP(nn.Module):
    def __init__(self, config):
        super(SimpleNetworkPP, self).__init__()
        self.setup_seed(0)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x, self.y0, self.t0 = None, None, None
        self.generate_x()
        self.initial_start()
        self.model_name = "SimpleNetworkPP"
        self.sig = nn.Tanh()
        self.fc1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.sig,
            'lin2': nn.Linear(100, 100),
            'sig2': self.sig,
            'lin3': nn.Linear(100, 100),
            'sig3': self.sig,
            'lin4': nn.Linear(100, 100),
            'sig4': self.sig,
            'lin5': nn.Linear(100, 1),
            # 'sig5': self.sig,
        }))

        self.fc2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.sig,
            'lin2': nn.Linear(100, 100),
            'sig2': self.sig,
            'lin3': nn.Linear(100, 100),
            'sig3': self.sig,
            'lin4': nn.Linear(100, 100),
            'sig4': self.sig,
            'lin5': nn.Linear(100, 1),
            # 'sig5': self.sig,
        }))

    def forward(self, inputs):
        u_old, v_old = torch.chunk(inputs, 2, 1)
        u_new = self.fc1(u_old)
        v_new = self.fc2(v_old)
        outputs = torch.cat((u_new, v_new), 1)
        return outputs

    def generate_x(self):
        # lb_row = np.repeat(self.config.lb, 2).reshape([1, -1])
        # ub_row = np.repeat(self.config.ub, 2).reshape([1, -1])
        # x = lb_row + (ub_row - lb_row) * lhs(1, self.config.N)
        # x = sorted(x, key=lambda xx: xx[0])
        x = [[i*0.001, i*0.001] for i in range(self.config.N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        self.x = torch.Tensor(x).float().to(self.device)

    def initial_start(self):
        self.t0 = torch.Tensor(np.asarray([-1.0, -1.0]).reshape([1, -1])).float().to(self.device)
        # self.y0 = torch.Tensor(np.asarray([0.0, 0.0]).reshape([1, -1])).float().to(self.device)
        self.y0 = torch.Tensor(np.asarray([self.config.U_start, self.config.V_start]).reshape([1, -1])).float().to(self.device)

    def loss(self):
        y = self.decode_y(self.forward(self.x))
        # print("y.shape:", y[:, 0:1].shape)
        # print(y[:, 0:1])
        # print("self.x.shape:", self.x[:, 0:1].shape)
        # y_t = torch.gradient(y, spacing=(self.x,))[0]  # y_t = tf.gradients(y, t)[0]
        u = y[:, 0:1]
        v = y[:, 1:2]

        u_t = torch.gradient(y[:, 0:1].reshape([self.config.N]), spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[0]  # u_t = y_t[:, 0:1]
        v_t = torch.gradient(y[:, 1:2].reshape([self.config.N]), spacing=(self.decode_t(self.x)[:, 1:2].reshape([self.config.N]),))[0]  # y_t[:, 1:2]
        u_t = u_t.reshape([self.config.N, 1])
        v_t = v_t.reshape([self.config.N, 1])
        # print(y_t,u_t)
        f_u = u_t - (self.config.alpha - self.config.gamma * v) * u  # nn model
        f_v = v_t - (-self.config.beta + self.config.e * self.config.gamma * u) * v  # nn model
        # f_u = u_t - torch.cos(self.decode_t(self.x)[:, 0:1])
        # f_v = v_t - torch.cos(self.decode_t(self.x)[:, 1:2])
        f_y = torch.cat((f_u, f_v), 1)
        y0_pred = self.decode_y(self.forward(self.encode_y(self.t0)))
        print_flag = False  # True # False
        if print_flag:
            print("u=", u.shape, u, "v=", v.shape, v, "t=", (self.decode_t(self.x)[:, 0:1]).shape, self.decode_t(self.x)[:, 0:1])
            print("u_t=", u_t, "shape=", u_t.shape, "u_t_minus=", (self.config.alpha - self.config.gamma * v) * u,
                  "shape=", ((self.config.alpha - self.config.gamma * v) * u).shape)
            print("v_t=", v_t, "shape=", v_t.shape, "v_t_minus=",
                  (-self.config.beta + self.config.e * self.config.gamma * u) * v, "shape=",
                  ((-self.config.beta + self.config.e * self.config.gamma * u) * v).shape)
            print("self.t0 =", self.t0, "self.encode_y(self.t0)=", self.encode_y(self.t0), "self.y0 =", self.y0,
                  "y0_pred =", y0_pred, "self.y0 - y0_pred =", self.y0 - y0_pred)

        # loss_2_weights = [[i, i] for i in range(20000)]  # toy
        # loss_2_weights = np.asarray(loss_2_weights)
        # loss_2_weights = torch.Tensor(loss_2_weights).float().to(self.device)

        loss_1 = torch.mean(torch.square(self.y0 - y0_pred))
        loss_2 = torch.mean(torch.square(f_y))  # + torch.var(torch.square(f_y))
        # loss_3 = torch.mean(torch.square(1/(u*u+v*v)))
        # loss_3 = torch.mean(torch.square(1.0 / u - (self.config.e * self.config.gamma / self.config.beta))) + \
        #     torch.mean(torch.square(1.0 / v - (self.config.gamma / self.config.alpha)))
        loss = loss_1 + loss_2
        # if loss < 2.0:
        #     f_y_square_pure = torch.square(f_y).cpu().detach().numpy()
        #     for i in range(20000):
        #         print(i, f_y_square_pure[i])

        return loss, [loss_1, loss_2]
        # return torch.mean(torch.square(y_hat - y))
        # return F.mse_loss(torch.cat((u_hat, v_hat), 1), torch.cat((u, v), 1))
        # return torch.abs(u_hat - u) + torch.abs(v_hat - v)  # F.mse_loss(x_hat, x) + beta * self.kl_div(rho)

    def encode_y(self, num):
        return num
        # return (num - self.config.y_lb) / (self.config.y_ub - self.config.y_lb) * 2.0 - 1.0

    def decode_y(self, num):
        return num
        # return self.config.y_lb + (num + 1.0) / 2.0 * (self.config.y_ub - self.config.y_lb)

    def encode_t(self, num):
        return (num - self.config.lb) / (self.config.ub - self.config.lb) * 2.0 - 1.0

    def decode_t(self, num):
        return self.config.lb + (num + 1.0) / 2.0 * (self.config.ub - self.config.lb)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True


class SimpleNetworkSIS(nn.Module):
    def __init__(self, config):
        super(SimpleNetworkSIS, self).__init__()
        self.setup_seed(0)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x, self.y0, self.t0 = None, None, None
        self.generate_x()
        # self.optimizer = optim.LBFGS(self.parameters(), lr=0.001, max_iter=5000, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        self.initial_start()
        self.model_name = "SimpleNetworkSIS"
        self.sig = nn.Tanh()
        self.fc1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.sig,
            'lin2': nn.Linear(100, 100),
            'sig2': self.sig,
            'lin3': nn.Linear(100, 100),
            'sig3': self.sig,
            'lin4': nn.Linear(100, 100),
            'sig4': self.sig,
            'lin5': nn.Linear(100, 1),
            # 'sig5': self.sig,
        }))

        self.fc2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.sig,
            'lin2': nn.Linear(100, 100),
            'sig2': self.sig,
            'lin3': nn.Linear(100, 100),
            'sig3': self.sig,
            'lin4': nn.Linear(100, 100),
            'sig4': self.sig,
            'lin5': nn.Linear(100, 1),
            # 'sig5': self.sig,
        }))

        self.fc3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.sig,
            'lin2': nn.Linear(100, 100),
            'sig2': self.sig,
            'lin3': nn.Linear(100, 100),
            'sig3': self.sig,
            'lin4': nn.Linear(100, 100),
            'sig4': self.sig,
            'lin5': nn.Linear(100, 1),
            # 'sig5': self.sig,
        }))

    def forward(self, inputs):
        s_old, i_old, r_old = torch.chunk(inputs, 3, 1)
        s_new = self.fc1(s_old)
        i_new = self.fc2(i_old)
        r_new = self.fc2(r_old)
        outputs = torch.cat((s_new, i_new, r_new), 1)
        return outputs

    def generate_x(self):
        # lb_row = np.repeat(self.config.lb, 2).reshape([1, -1])
        # ub_row = np.repeat(self.config.ub, 2).reshape([1, -1])
        # x = lb_row + (ub_row - lb_row) * lhs(1, self.config.N)
        # x = sorted(x, key=lambda xx: xx[0])
        x = [[i*self.config.T_unit, i*self.config.T_unit, i*self.config.T_unit] for i in range(self.config.N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        self.x = torch.Tensor(x).float().to(self.device)

    def initial_start(self):
        self.t0 = torch.Tensor(np.asarray([-1.0, -1.0, -1.0]).reshape([1, -1])).float().to(self.device)
        self.y0 = torch.Tensor(np.asarray([self.config.S_start, self.config.I_start, self.config.R_start]).reshape([1, -1])).float().to(self.device)
        self.tend = torch.Tensor(np.asarray([1.0, 1.0, 1.0]).reshape([1, -1])).float().to(self.device)

    def loss(self):
        y = self.decode_y(self.forward(self.x))
        # print("y.shape:", y[:, 0:1].shape)
        # print(y[:, 0:1])
        # print("self.x.shape:", self.x[:, 0:1].shape)
        # y_t = torch.gradient(y, spacing=(self.x,))[0]  # y_t = tf.gradients(y, t)[0]
        s = y[:, 0:1]
        i = y[:, 1:2]
        r = y[:, 2:3]

        s_t = torch.gradient(y[:, 0:1].reshape([self.config.N]), spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[0]
        i_t = torch.gradient(y[:, 1:2].reshape([self.config.N]), spacing=(self.decode_t(self.x)[:, 1:2].reshape([self.config.N]),))[0]
        r_t = torch.gradient(y[:, 2:3].reshape([self.config.N]), spacing=(self.decode_t(self.x)[:, 1:2].reshape([self.config.N]),))[0]

        s_t = s_t.reshape([self.config.N, 1])
        i_t = i_t.reshape([self.config.N, 1])
        r_t = r_t.reshape([self.config.N, 1])
        # print(y_t,u_t)
        f_s = s_t - (- self.config.beta * s * i)
        f_i = i_t - (self.config.beta * s * i - self.config.gamma * i)
        f_r = r_t - (self.config.gamma * i)
        # f_u = u_t - (self.config.alpha - self.config.gamma * v) * u
        # f_v = v_t - (-self.config.beta + self.config.e * self.config.gamma * u) * v
        # f_u = u_t - torch.cos(self.decode_t(self.x)[:, 0:1])  # toy
        # f_v = v_t - torch.cos(self.decode_t(self.x)[:, 1:2])  # toy
        f_y = torch.cat((f_s, f_i, f_r), 1)
        y0_pred = self.decode_y(self.forward(self.encode_y(self.t0)))
        yend_pred = self.decode_y(self.forward(self.encode_y(self.tend)))
        # print_flag = False  # True # False
        # if print_flag:
        #     print("u=", u.shape, u, "v=", v.shape, v, "t=", (self.decode_t(self.x)[:, 0:1]).shape, self.decode_t(self.x)[:, 0:1])
        #     print("u_t=", u_t, "shape=", u_t.shape, "u_t_minus=", (self.config.alpha - self.config.gamma * v) * u,
        #           "shape=", ((self.config.alpha - self.config.gamma * v) * u).shape)
        #     print("v_t=", v_t, "shape=", v_t.shape, "v_t_minus=",
        #           (-self.config.beta + self.config.e * self.config.gamma * u) * v, "shape=",
        #           ((-self.config.beta + self.config.e * self.config.gamma * u) * v).shape)
        #     print("self.t0 =", self.t0, "self.encode_y(self.t0)=", self.encode_y(self.t0), "self.y0 =", self.y0,
        #           "y0_pred =", y0_pred, "self.y0 - y0_pred =", self.y0 - y0_pred)

        # loss_2_weights = [[i, i] for i in range(20000)]  # toy
        # loss_2_weights = np.asarray(loss_2_weights)
        # loss_2_weights = torch.Tensor(loss_2_weights).float().to(self.device)

        loss_1 = torch.mean(torch.square(self.y0 - y0_pred))
        loss_2 = torch.mean(torch.square(f_y))  # + torch.var(torch.square(f_y))
        # loss_3 = torch.mean(torch.square(1/(u*u+v*v)))
        # loss_3 = torch.mean(torch.square(1.0 / u - (self.config.e * self.config.gamma / self.config.beta))) + \
        #     torch.mean(torch.square(1.0 / v - (self.config.gamma / self.config.alpha)))
        # loss_3 = 10 * (torch.abs(torch.sum(y0_pred) - self.config.SIR_sum) + torch.abs(torch.sum(yend_pred) - self.config.SIR_sum))
        loss = loss_1 + loss_2
        # if loss < 2.0:
        #     f_y_square_pure = torch.square(f_y).cpu().detach().numpy()
        #     for i in range(20000):
        #         print(i, f_y_square_pure[i])

        return loss, [loss_1, loss_2, f_y.cpu().detach().numpy()]
        # return torch.mean(torch.square(y_hat - y))
        # return F.mse_loss(torch.cat((u_hat, v_hat), 1), torch.cat((u, v), 1))
        # return torch.abs(u_hat - u) + torch.abs(v_hat - v)  # F.mse_loss(x_hat, x) + beta * self.kl_div(rho)


    def encode_y(self, num):
        return num
        # return (num - self.config.y_lb) / (self.config.y_ub - self.config.y_lb) * 2.0 - 1.0

    def decode_y(self, num):
        return num
        # return self.config.y_lb + (num + 1.0) / 2.0 * (self.config.y_ub - self.config.y_lb)

    def encode_t(self, num):
        return (num - self.config.lb) / (self.config.ub - self.config.lb) * 2.0 - 1.0

    def decode_t(self, num):
        return self.config.lb + (num + 1.0) / 2.0 * (self.config.ub - self.config.lb)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
