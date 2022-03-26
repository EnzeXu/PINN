import torch
import time
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torchsummary import summary

from config import ConfigPP, ConfigSIS
from model import SimpleNetwork, SimpleNetworkPP, SimpleNetworkSIS


def train_pp(model, args, config, now_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model.train()
    model_save_path_last = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.alpha}_{config.beta}_{config.gamma}_{config.e}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.alpha}_{config.beta}_{config.gamma}_{config.e}_{now_string}_best.pt"
    print("using " + str(device))
    print("epoch = {}".format(args.epoch))
    print("epoch_step = {}".format(args.epoch_step))
    print("model_name = {}".format(model.model_name))
    print("now_string = {}".format(now_string))
    print("model_save_path_last = {}".format(model_save_path_last))
    print("model_save_path_best = {}".format(model_save_path_best))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch_step = args.epoch_step
    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        inputs = model.x
        outputs = model(inputs)
        # u_hat, v_hat = torch.chunk(outputs, 2, 1)
        loss, loss_list = model.loss()
        loss_1, loss_2, = loss_list[0], loss_list[1]
        loss.backward()
        optimizer.step()
        best_loss = 999999
        if epoch % epoch_step == 0:
            print("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} Loss_1:{3:.6f} Loss_2:{4:.6f} Time:{5:.6f}s".format(epoch, args.epoch, loss.item(), loss_1.item(), loss_2.item(), time.time() - start_time))
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
                torch.save(
                    {
                        'epoch': args.epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, model_save_path_best)
        if epoch % args.save_step == 0:
            test_pp(model, args, config, now_string, True)


def test_pp(model, args, config, now_string, show_flag=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model_save_path = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.alpha}_{config.beta}_{config.gamma}_{config.e}_{now_string}_last.pt"
    model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
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
    u = model.decode_y(u)
    v = model.decode_y(v)
    u = [item[0] for item in u.cpu().detach().numpy()]
    v = [item[0] for item in v.cpu().detach().numpy()]
    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]
    pairs = [[uu, vv, xx] for uu, vv, xx in zip(u, v, x)]
    pairs.sort(key=lambda xx: xx[2])
    u = [item[0] for item in pairs]
    v = [item[1] for item in pairs]
    x = [item[2] for item in pairs]
    print("u=", u[:10], "...", u[-10:])
    print("v=", v[:10], "...", v[-10:])
    print("x=", x[:10], "...", x[-10:])
    plt.plot(x, u, marker='.', markersize=0.2, linewidth=0.1, c="b")
    plt.plot(x, v, marker='.', markersize=0.2, linewidth=0.1, c="r")
    figure_save_path = f"{args.main_path}/figure/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.alpha}_{config.beta}_{config.gamma}_{config.e}_{now_string}_{int(time.time())}.png"
    plt.savefig(figure_save_path, dpi=300)
    if show_flag:
        plt.show()
    plt.clf()


def train_sis(model, args, config, now_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model.train()
    model_save_path_last = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_last.pt"
    model_save_path_best = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_best.pt"
    print("using " + str(device))
    print("epoch = {}".format(args.epoch))
    print("epoch_step = {}".format(args.epoch_step))
    print("model_name = {}".format(model.model_name))
    print("now_string = {}".format(now_string))
    print("model_save_path_last = {}".format(model_save_path_last))
    print("model_save_path_best = {}".format(model_save_path_best))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.LBFGS(model.parameters(), lr=args.lr, max_iter=5000, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100,
    #       line_search_fn=None)
    epoch_step = args.epoch_step
    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        inputs = model.x
        outputs = model(inputs)
        # u_hat, v_hat = torch.chunk(outputs, 2, 1)
        loss, loss_list = model.loss()
        loss_1, loss_2, = loss_list[0], loss_list[1]
        loss.backward()
        optimizer.step()
        best_loss = 999999
        if epoch % epoch_step == 0:
            print("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} Loss_1:{3:.6f} Loss_2:{4:.6f} Time:{5:.6f}s".format(epoch, args.epoch, loss.item(), loss_1.item(), loss_2.item(), time.time() - start_time))
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
                torch.save(
                    {
                        'epoch': args.epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, model_save_path_best)
        if epoch % args.save_step == 0:
            test_sis(model, args, config, now_string, True)


def test_sis(model, args, config, now_string, show_flag=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_framework(config).to(device)
    model_save_path = f"{args.main_path}/train/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_last.pt"
    model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
    model.eval()
    print("Testing & drawing...")
    t = model.x
    y = model(t)
    y0_pred = model(model.t0)
    # print("t=", t)
    # print("t0=", model.t0)
    # print("y=", y)
    # print("y0_pred=", y0_pred)
    s, i, r = torch.chunk(y, 3, 1)
    # print("u=", u)
    # print("v=", v)
    s = model.decode_y(s)
    i = model.decode_y(i)
    r = model.decode_y(r)
    s = [item[0] for item in s.cpu().detach().numpy()]
    i = [item[0] for item in i.cpu().detach().numpy()]
    r = [item[0] for item in r.cpu().detach().numpy()]
    x = [item[0] for item in model.decode_t(t).cpu().detach().numpy()]
    pairs = [[ss, ii, rr, xx] for ss, ii, rr, xx in zip(s, i, r, x)]
    pairs.sort(key=lambda xx: xx[-1])
    s = [item[0] for item in pairs]
    i = [item[1] for item in pairs]
    r = [item[2] for item in pairs]
    x = [item[3] for item in pairs]
    print("s=", s[:10], "...", s[-10:])
    print("i=", i[:10], "...", i[-10:])
    print("r=", r[:10], "...", r[-10:])
    print("x=", x[:10], "...", x[-10:])
    loss, loss_list = model.loss()
    print(loss_list[2])
    plt.plot(x, s, marker='.', markersize=0.2, linewidth=0.1, c="b")
    plt.plot(x, i, marker='.', markersize=0.2, linewidth=0.1, c="r")
    plt.plot(x, r, marker='.', markersize=0.2, linewidth=0.1, c="g")
    figure_save_path = f"{args.main_path}/figure/{model.model_name}_{args.epoch}_{args.epoch_step}_{args.lr}_{config.beta}_{config.gamma}_{now_string}_{int(time.time())}.png"
    plt.savefig(figure_save_path, dpi=300)
    if show_flag:
        plt.show()
    plt.clf()


def summary_model(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNetworkPP().to(device)
    model_save_path = f"{args.main_path}/train/model_{args.epoch}_{args.epoch_step}_{args.lr}_{config.alpha}_{config.beta}_{config.gamma}_{config.e}_best.pt"
    model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
    summary(model, (1,1, 2))


def get_now_string():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))


def run_pp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=10, help="save_step")
    args = parser.parse_args()
    now_string = get_now_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigPP
    model = SimpleNetworkPP(config).to(device)
    train_pp(model, args, config, now_string)
    model = SimpleNetworkPP(config).to(device)
    test_pp(model, args, config, now_string)


def run_sis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=10, help="epoch_step")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.001')
    parser.add_argument("--main_path", default=".", help="main_path")
    parser.add_argument("--save_step", type=int, default=100, help="save_step")
    args = parser.parse_args()
    now_string = get_now_string()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigSIS
    model = SimpleNetworkSIS(config).to(device)
    train_sis(model, args, config, now_string)
    model = SimpleNetworkSIS(config).to(device)
    test_sis(model, args, config, now_string)


if __name__ == "__main__":
    run_sis()

    # summary_model(ConfigPP)
    pass
