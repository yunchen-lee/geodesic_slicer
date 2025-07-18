# heat_main.py
# 替代 Neural Slicer 的 main.py，使用 heat field h(x) 為初始場，經 SIREN 擬合後訓練

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd 
# import torch.nn.autograd as autograd
import torch.nn as nn
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import models.SIREN as SIREN
# from utils.logger import Logger  # optional: 可留可拔

import numpy as np
# import argparse
import os.path as osp         # Python 內建：操作檔案路徑
from tqdm import tqdm                  # 進度條顯示


def import_modules():
    print(np.__version__)
    print("modules imported.")

def load_data():
    elem_center = np.loadtxt('data/0717_elem_center.csv', delimiter=",")  # [N, 3]
    h_init = np.loadtxt('data/0717_heat_value.csv', delimiter=",") # [N, 1]
    adjacent_edges = np.loadtxt('data/0717_adjacent_edges.csv', delimiter=",") # [N, 2]
    return elem_center, h_init, adjacent_edges


def layer_uniformity_loss(h_pred: torch.Tensor,
                          elem_center: torch.Tensor,
                          adj_i: torch.Tensor,
                          adj_j: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:

    # 差分近似 ∇h ⋅ edge
    dh = h_pred[adj_i] - h_pred[adj_j]                      # shape (M, 1)
    dx = elem_center[adj_i] - elem_center[adj_j]            # shape (M, 3)
    dist = torch.norm(dx, dim=1, keepdim=True) + eps        # shape (M, 1)
    grad_along_edge = dh / dist                             # shape (M, 1)，邊方向的梯度近似

    # 統計量：梯度量值的變異性
    grad_mag = grad_along_edge.abs()                        # shape (M, 1)
    mean_grad = grad_mag.mean()
    loss = ((grad_mag - mean_grad) ** 2).mean()             # 方差損失

    return loss

def directional_consistency_loss(h_pred: torch.Tensor,
                                 elem_center: torch.Tensor,
                                 adj_i: torch.Tensor,
                                 adj_j: torch.Tensor,
                                 eps: float = 1e-8) -> torch.Tensor:
    # 反向圖需要 ∇h(x)
    gradients = autograd.grad(
        outputs=h_pred.sum(), 
        inputs=elem_center, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]  # shape: (N, 3)

    grad_unit = F.normalize(gradients + eps, dim=1)  # 單位化梯度，防止除 0

    grad_i = grad_unit[adj_i]  # shape (M, 3)
    grad_j = grad_unit[adj_j]  # shape (M, 3)

    cos_sim = (grad_i * grad_j).sum(dim=1, keepdim=True)  # 內積 → cosθ
    loss = (1.0 - cos_sim).mean()  # 越接近 1，方向越一致

    return loss


def train():

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device setting: {device}")

    # load mesh verties and heat value from grasshopper
    elem_center, h_init, adjacent_edges = load_data()

    # elem_center normalize
    elem_min = elem_center.min(axis=0)
    elem_max = elem_center.max(axis=0)
    range_val = elem_max - elem_min
    range_val[range_val == 0] = 1.0
    elem_center_normalized = 2 * (elem_center - elem_min) / range_val - 1

    # heat_value normalize
    h_min = h_init.min()
    h_max = h_init.max()
    h_range = h_max - h_min
    if h_range == 0:
        h_range = 1.0
        print("reset h_init range to 1")
    h_init_normalized = (h_init - h_min) / h_range

    # adjacent_edges 
    adj_i = torch.from_numpy(adjacent_edges[:, 0].astype(np.int64)).long().cuda()
    adj_j = torch.from_numpy(adjacent_edges[:, 1].astype(np.int64)).long().cuda()



    # ====================================================================
    # Phase 1: 
    # ====================================================================
    # set traning parameters
    nstep = 1000    # number of training steps
    lrate = 1e-4    # learning rate
    min_lr = 1e-7   # min learning rate
    factor = 0.9
    patience = 20
    cooldown = 200

    # use SIREN train h(x) to a cts, diff function for nn.
    net = SIREN.SimpleSirenNet(
        dim_in=3, 
        dim_hidden=512,
        dim_out=1, 
        num_layers=5,
        w0_initial=30,
        w0=1
    ).to(device)

    elem_center_tensor = torch.from_numpy(elem_center_normalized).float().cuda()
    # elem_center_tensor.requires_grad_(True) # # try grad in phase 1
    h_init_tensor = torch.from_numpy(h_init_normalized).float().cuda()
    net = net.float().cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                        min_lr=min_lr, 
                        factor=factor,
                        patience=patience, 
                        cooldown=cooldown)
    criterion = nn.MSELoss() 

    # Phase 1: 選擇是否先擬合 h_init
    print("Fitting to heat field as initialization...")
    for step in tqdm(range(nstep)):
        # net.train()
        optimizer.zero_grad()
        h_pred = net(elem_center_tensor)

        loss = F.mse_loss(h_pred, h_init_tensor.view(-1, 1))
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)   #

        if step % 100 == 0:
            tqdm.write(f'Epoch [{step}/{nstep}], Loss: {loss.item():.8f}')

    print("Total parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # ====================================================================
    # Phase 2: 
    # ====================================================================
    
    loss_record = {
    "step": [],
    "base": [],
    "uniform": [],
    "direction": [],
    "total": []
    }

    print("\nOptimizing with custom loss (e.g., smoothness)...")
    elem_center_tensor_optim = elem_center_tensor.detach().clone()
    elem_center_tensor_optim.requires_grad_(True)


    # set traning parameters
    nstep_optim = 1000    # number of training steps
    lrate_optim = 1e-4    # learning rate
    min_lr_optim = 1e-7   # min learning rate
    factor_optim = 0.9
    patience_optim = 20
    cooldown_optim = 200
    w_smoothness = 0.01
    w_uniform = 0.1
    w_directional = 0.1


    optimizer_optim = torch.optim.Adam(net.parameters(), lr = lrate_optim)
    scheduler_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_optim, 
                                                        min_lr = min_lr_optim,
                                                        factor = factor_optim,
                                                        patience = patience_optim,
                                                        cooldown = cooldown_optim)

    
    for step in tqdm(range(nstep_optim)):
        optimizer_optim.zero_grad()
        h_pred_optim = net(elem_center_tensor_optim)

        base_loss = F.mse_loss(h_pred_optim, h_init_tensor.view(-1, 1))

        uniform_loss = layer_uniformity_loss(h_pred_optim, elem_center_tensor_optim, adj_i, adj_j)
        directional_loss = directional_consistency_loss(h_pred_optim, elem_center_tensor_optim, adj_i, adj_j)
        total_loss = base_loss + w_uniform * uniform_loss + w_directional * directional_loss

        total_loss.backward()
        optimizer_optim.step()
        scheduler_optim.step(total_loss)

        if step % 50 == 0:
             tqdm.write(f'Optimize Epoch [{step}/{nstep_optim}], Total Loss: {total_loss.item():.8f}, Base Loss: {base_loss.item():.8f}, Uniform Loss: {uniform_loss.item():.8f}, Direction Loss: {directional_loss.item():.8f}, LR: {optimizer_optim.param_groups[0]["lr"]:.2e}')
        # 記錄 loss
        if step % 10 == 0:
            loss_record["step"].append(step)
            loss_record["base"].append(base_loss.item())
            # loss_record["smooth"].append(smoothness_loss.item())
            loss_record["direction"].append(directional_loss.item())
            loss_record["uniform"].append(uniform_loss.item())
            loss_record["total"].append(total_loss.item())

    # After optimization, visualize the new h_pred
    h_pred_optimized = net(elem_center_tensor).detach().cpu().numpy().flatten() # Use original elem_center_tensor for final prediction
    h_true_np = h_init_tensor.cpu().numpy().flatten()


    # -----------------------------------
    # // plt for phase 2 //

    # Plotting the loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_record["step"], loss_record["total"], label='Total Loss', color='blue')
    plt.plot(loss_record["step"], loss_record["base"], label='Base Loss (MSE)', color='green')
    # plt.plot(loss_record["step"], loss_record["smooth"], label='Smoothness Loss', color='orange')
    plt.plot(loss_record["step"], loss_record["uniform"], label='Uniform Loss', color='orange')
    plt.plot(loss_record["step"], loss_record["direction"], label='Direction Loss', color='red')


    plt.xlabel("Training Step")
    plt.ylabel("Loss Value")
    plt.title("Loss Decrease Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

    # ====================================================================
    # // export csv
    # ====================================================================

    # export elem_center (denormalized，input elem_center original data）
    elem_center_out = elem_center  # [N, 3] numpy array from mesh.csv

    # denormalized h_pred
    h_pred_denorm = h_pred_optimized * h_range + h_min
    # h_pred_denorm = h_pred_np * h_range + h_min
    h_pred_out = h_pred_denorm.reshape(-1, 1)

    # 輸出為 [h_pred]
    output_data = np.hstack((h_pred_out))
    np.savetxt("output_heat_optimized.csv", output_data, delimiter=",", fmt="%.6f",
            header="h_pred", comments='')
    print("✔ Exported denormalized heat field：output_heat_optimized.csv")

    print("program completed.")

if __name__ == '__main__':
    train()
