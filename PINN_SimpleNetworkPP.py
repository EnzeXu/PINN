import torch
import time
import random
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from collections import OrderedDict
from torch.backends import cudnn


class ConfigPP:
    T = 20.0
    T_unit = 0.001
    N = int(T / T_unit)
    U_start = 10.0
    V_start = 5.0
    alpha = 1.0
    beta = 3.0
    gamma = 0.3
    e = 0.333
    ub = T
    lb = 0.0


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
        self.act = nn.Tanh()

        # Design D
        self.fc_x1_0_1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.act,
        }))

        self.fc_x2_0_1 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(1, 100),
            'sig1': self.act,
        }))

        self.fc_x1_1_2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(200, 100),
            'sig1': self.act,
        }))

        self.fc_x2_1_2 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(200, 100),
            'sig1': self.act,
        }))

        self.fc_x1_2_3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(200, 100),
            'sig1': self.act,
        }))

        self.fc_x2_2_3 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(200, 100),
            'sig1': self.act,
        }))

        self.fc_x1_3_4 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(100, 1),
        }))

        self.fc_x2_3_4 = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(100, 1),
        }))

    def forward(self, inputs):
        # Design D
        x1, x2 = torch.chunk(inputs, 2, 1)
        x1_1_output = self.fc_x1_0_1(x1)
        x2_1_output = self.fc_x2_0_1(x2)

        x1_2_input = torch.cat((x1_1_output, x2_1_output), 1)
        x2_2_input = torch.cat((x1_1_output, x2_1_output), 1)

        x1_2_output = self.fc_x1_1_2(x1_2_input)
        x2_2_output = self.fc_x2_1_2(x2_2_input)

        x1_3_input = torch.cat((x1_2_output, x2_2_output), 1)
        x2_3_input = torch.cat((x1_2_output, x2_2_output), 1)

        x1_3_output = self.fc_x1_2_3(x1_3_input)
        x2_3_output = self.fc_x2_2_3(x2_3_input)

        x1_new = self.fc_x1_3_4(x1_3_output)
        x2_new = self.fc_x2_3_4(x2_3_output)

        outputs = torch.cat((x1_new, x2_new), 1)
        return outputs

    def generate_x(self):
        # lb_row = np.repeat(self.config.lb, 2).reshape([1, -1])
        # ub_row = np.repeat(self.config.ub, 2).reshape([1, -1])
        # x = lb_row + (ub_row - lb_row) * lhs(1, self.config.N)
        # x = sorted(x, key=lambda xx: xx[0])
        x = [[i * self.config.T_unit, i * self.config.T_unit] for i in range(self.config.N)]  # toy
        x = np.asarray(x)
        x = self.encode_t(x)
        self.x = torch.Tensor(x).float().to(self.device)

    def initial_start(self):
        self.t0 = torch.Tensor(np.asarray([-1.0, -1.0]).reshape([1, -1])).float().to(self.device)
        # self.y0 = torch.Tensor(np.asarray([0.0, 0.0]).reshape([1, -1])).float().to(self.device)
        self.y0 = torch.Tensor(np.asarray([self.config.U_start, self.config.V_start]).reshape([1, -1])).float().to(
            self.device)

    def loss(self):
        self.eval()
        y = self.forward(self.x)
        y0_pred = self.forward(self.t0)
        u = y[:, 0:1]
        v = y[:, 1:2]

        u_t = torch.gradient(y[:, 0:1].reshape([self.config.N]),
                             spacing=(self.decode_t(self.x)[:, 0:1].reshape([self.config.N]),))[0]  # u_t = y_t[:, 0:1]
        v_t = torch.gradient(y[:, 1:2].reshape([self.config.N]),
                             spacing=(self.decode_t(self.x)[:, 1:2].reshape([self.config.N]),))[0]  # y_t[:, 1:2]
        u_t = u_t.reshape([self.config.N, 1])
        v_t = v_t.reshape([self.config.N, 1])
        # print(y_t,u_t)
        f_u = u_t - (self.config.alpha - self.config.gamma * v) * u  # nn model
        f_v = v_t - (-self.config.beta + self.config.e * self.config.gamma * u) * v  # nn model
        # f_u = u_t - torch.cos(self.decode_t(self.x)[:, 0:1])
        # f_v = v_t - torch.cos(self.decode_t(self.x)[:, 1:2])
        f_y = torch.cat((f_u, f_v), 1)

        # L2 norm
        loss_norm = torch.nn.MSELoss().to(self.device)
        zeros_1D = torch.Tensor([[0.0]] * self.config.N).to(self.device)
        zeros_2D = torch.Tensor([[0.0, 0.0]] * self.config.N).to(self.device)
        loss_1 = loss_norm(y0_pred, self.y0)
        loss_2 = loss_norm(f_y, zeros_2D)  # + torch.var(torch.square(f_y))
        loss_3 = loss_norm((0.1 / (u * u + v * v + 1e-12)), zeros_1D) + loss_norm(torch.abs(u), u) + loss_norm(
            torch.abs(v), v)

        # loss_3 = torch.mean(torch.square(1.0 / u - (self.config.e * self.config.gamma / self.config.beta))) + \
        #     torch.mean(torch.square(1.0 / v - (self.config.gamma / self.config.alpha)))
        # loss_3 = torch.mean(torch.square((torch.abs(u) - u)))

        # loss_3 = torch.mean(torch.square((1/(u * u+ 1e-12)))) + torch.mean(torch.square((1/(v * v + 1e-12)))) + torch.mean((torch.abs(u) - u)) + torch.mean((torch.abs(v) - v))

        # loss_3 = torch.mean(torch.square((1/(u * u + v * v)))) + torch.mean((torch.abs(u) - u)) + torch.mean((torch.abs(v) - v))  # torch.mean(torch.square((1/(u * u + v * v + 1e-12)))) + torch.mean((torch.abs(u) - u)) + torch.mean((torch.abs(v) - v))
        # loss_4 = torch.mean(0.001 / (torch.abs(u_t)+1e-8)) + torch.mean(0.001 / (torch.abs(v_t)+1e-8))
        # loss_5 = torch.mean(0.001 / (torch.abs(u_tt)+1e-8)) + torch.mean(0.001 / (torch.abs(v_tt)+1e-8))
        loss = (loss_1 + loss_2 + loss_3)  # + loss_4 + loss_5) / 1e5
        # if loss < 2.0:
        #     f_y_square_pure = torch.square(f_y).cpu().detach().numpy()
        #     for i in range(20000):
        #         print(i, f_y_square_pure[i])
        self.train()
        return loss, [loss_1, loss_2, loss_3]
        # return torch.mean(torch.square(y_hat - y))
        # return F.mse_loss(torch.cat((u_hat, v_hat), 1), torch.cat((u, v), 1))
        # return torch.abs(u_hat - u) + torch.abs(v_hat - v)  # F.mse_loss(x_hat, x) + beta * self.kl_div(rho)

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


def get_now_string():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


def draw_loss(loss_list, show_flag=True):
    map = np.asarray([[loss] for loss in loss_list])
    plt.plot(map)
    if show_flag:
        plt.show()
    plt.clf()


def train_pp(model, args, config, now_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model.train()
    model_save_path_last = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_best.pt"
    loss_save_path = f"{args.main_path}/loss/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_loss_{args.epoch}.npy"
    print("using " + str(device))
    print("epoch = {}".format(args.epoch))
    print("epoch_step = {}".format(args.epoch_step))
    print("model_name = {}".format(model.model_name))
    print("now_string = {}".format(now_string))
    print("model_save_path_last = {}".format(model_save_path_last))
    print("model_save_path_best = {}".format(model_save_path_best))
    print("loss_save_path = {}".format(loss_save_path))
    print("args = {}".format({item[0]: item[1] for item in args.__dict__.items() if item[0][0] != "_"}))
    print("config = {}".format({item[0]: item[1] for item in config.__dict__.items() if item[0][0] != "_"}))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    initial_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 10000 + 1))  # decade
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size=10000) # cyclic
    epoch_step = args.epoch_step
    start_time = time.time()
    start_time_0 = start_time
    best_loss = 999999
    now_time = 0
    loss_record = []

    def closure():
        optimizer.zero_grad()
        inputs = model.x
        outputs = model(inputs)
        loss, loss_list = model.loss()
        return loss

    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        inputs = model.x
        outputs = model(inputs)
        loss, loss_list = model.loss()

        loss_1, loss_2, loss_3 = loss_list[0], loss_list[1], loss_list[2]
        loss.backward()
        # options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        # _loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(closure)
        optimizer.step()
        scheduler.step()
        loss_record.append(float(loss.item()))
        if epoch % epoch_step == 0:
            now_time = time.time()
            print(
                "Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} Loss_1:{3:.6f} Loss_2:{4:.6f} Loss_3: {5:.6f} Lr:{6:.6f} Time:{7:.6f}s ({8:.2f}min in total)".format(
                    epoch, args.epoch, loss.item(), loss_1.item(), loss_2.item(), loss_3.item(),
                    optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0))
            start_time = time.time()
            torch.save(
                {
                    'epoch': args.epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, model_save_path_last)
            # print(inputs.shape)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        'epoch': args.epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, model_save_path_best)
        if epoch % args.save_step == 0:
            test_pp(model, args, config, now_string, False)
            draw_loss(np.asarray(loss_record), False)

    num_parameter = -1  # get_model_parameters(model, config)
    best_loss = best_loss
    time_cost = (now_time - start_time_0) / 60.0
    loss_record = np.asarray(loss_record)
    np.save(loss_save_path, loss_record)
    # draw_loss(loss_record)
    return [num_parameter, best_loss, time_cost, loss_record]


def test_pp(model, args, config, now_string, show_flag=True):
    model_save_path = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_last.pt"
    predict_save_path = f"{args.main_path}/predict/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_predict.pkl"
    # model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
    model.eval()
    print("Testing & drawing...")
    t = model.x
    y = model(t)
    y0_pred = model(model.t0)
    # print("t=", t)
    # print("t0=", model.t0)
    # print("y=", y)
    # print("y0_pred=", y0_pred)
    u, v = torch.chunk(y, 2, 1)
    # print("u=", u)
    # print("v=", v)
    u = [item[0] for item in u.cpu().detach().numpy()]
    v = [item[0] for item in v.cpu().detach().numpy()]
    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]
    pairs = [[uu, vv, xx] for uu, vv, xx in zip(u, v, x)]
    pairs.sort(key=lambda xx: xx[2])
    u = [item[0] for item in pairs]
    v = [item[1] for item in pairs]
    x = [item[2] for item in pairs]
    predict_dic = {
        "x": x,
        "x1": u,
        "x2": v
    }
    with open(predict_save_path, "wb") as f:
        pickle.dump(predict_dic, f)
    print("u=", u[:10], "...", u[-10:])
    print("v=", v[:10], "...", v[-10:])
    print("x=", x[:10], "...", x[-10:])
    plt.plot(x, u, marker='.', markersize=0.2, linewidth=0.1, c="b")
    plt.plot(x, v, marker='.', markersize=0.2, linewidth=0.1, c="r")
    figure_save_path = f"{args.main_path}/figure/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{now_string}_{int(time.time())}.png"
    plt.savefig(figure_save_path, dpi=300)
    if show_flag:
        plt.show()
    plt.clf()
    print("Saved as {}".format(figure_save_path))


class Args:
    epoch = 100000
    epoch_step = 1000
    lr = 0.01
    main_path = "."
    save_step = 10000


def run_pp(main_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100000, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=1000, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=10000, help="save_step")
    args = parser.parse_args()
    # args = Args
    if main_path:
        args.main_path = main_path
    if not os.path.exists("{}/train".format(args.main_path)):
        os.makedirs("{}/train".format(args.main_path))
    if not os.path.exists("{}/figure".format(args.main_path)):
        os.makedirs("{}/figure".format(args.main_path))
    if not os.path.exists("{}/loss".format(args.main_path)):
        os.makedirs("{}/loss".format(args.main_path))
    now_string = get_now_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigPP
    model = SimpleNetworkPP(config).to(device)
    res_list = train_pp(model, args, config, now_string)

    print("\n[Training Summary]")
    # print("num_parameter: {}".format(res_list[0]))
    print("best_loss: {}".format(res_list[1]))
    print("time_cost: {}".format(res_list[2]))


if __name__ == "__main__":
    run_pp()

