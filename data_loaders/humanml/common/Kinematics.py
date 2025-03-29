import torch
import torch.nn as nn
import numpy as np
import math
from data_loaders.humanml.common.rotation import qrot, quat_fk, q2cont6d, cont6d2q


def recover_root_rot_pos_this(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(r_rot_quat, r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


class InverseKinematics_hmlvec:
    def __init__(self, data, joints_num, skeleton, real_offsets, constraints, use_lbfgs=False):
       
        r_rot_quat, r_pos = recover_root_rot_pos_this(data)
        cont6d_params = data[..., 4 + (joints_num-1) * 3:].reshape(data.shape[:-1] + (joints_num, 6))
        self.cont6d_params = cont6d_params
        self.r_pos = r_pos
        self.r_rot_quat = r_rot_quat
        self.skeleton = skeleton
        self.offset = torch.Tensor(real_offsets)
        self.use_lbfgs = use_lbfgs

        self.r_pos.requires_grad_(True)
        self.cont6d_params.requires_grad_(True)
        self.r_rot_quat.requires_grad_(True)

        self.constraints = constraints
        if use_lbfgs:
            self.optimizer = torch.optim.LBFGS([self.cont6d_params, self.r_pos, self.r_rot_quat], max_iter=10,
                                                 lr=1e-2, line_search_fn=None)
        else:
            self.optimizer = torch.optim.Adam([self.cont6d_params, self.r_pos, self.r_rot_quat], lr=1e-3, betas=(0.9, 0.999))
            # self.optimizer = torch.optim.SparseAdam([self.cont6d_params, self.r_pos, self.r_rot_quat], lr=1e-3)
            

        self.crit = nn.MSELoss()

    def gmof(self, x, sigma):
        """
        Geman-McClure error function
        """
        x_squared = x ** 2 
        sigma_squared = sigma ** 2
        return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def step(self):
        if not self.use_lbfgs:
            self.optimizer.zero_grad()
            glb = self.forward(self.cont6d_params, self.r_pos, self.r_rot_quat)
            # loss = self.crit(glb, self.constraints)
            loss = self.gmof(glb-self.constraints, 100).sum(-1).sum(-1).sum(-1)
            loss.backward()
            self.optimizer.step()
            self.glb = glb
            return loss.item()
        else:
            def closure():
                self.optimizer.zero_grad()
                glb = self.forward(self.cont6d_params, self.r_pos, self.r_rot_quat)
                # loss = self.crit(glb, self.constraints) * (500**2)
                loss = self.gmof(glb-self.constraints, 100).sum(-1).sum(-1).sum(-1)
                loss.backward()
                print(loss.item())
                return loss
            self.optimizer.step(closure)

    
    def forward(self, cont6d_params, r_pos, r_rot_quat):
        
        positions = self.skeleton.forward_kinematics_real_cont6d(cont6d_params, r_pos, r_rot_quat, self.offset)

        return positions
    

class InverseKinematics_quats:
    def __init__(self, quats, pos, parents, constraints):
        # r_rot_quat, r_pos = recover_root_rot_pos_this(data)
        # cont6d_params = data[..., 4 + (joints_num-1) * 3:].reshape(data.shape[:-1] + (joints_num, 6))
        # self.cont6d_params = cont6d_params
        # self.r_pos = r_pos
        # self.r_rot_quat = r_rot_quat
        # self.skeleton = skeleton

        self.cont6d = q2cont6d(quats)
        self.pos = pos
        self.parents = parents
        # self.offset = torch.Tensor(real_offsets)

        self.cont6d.requires_grad_(True)
        # self.r_pos.requires_grad_(True)
   
        self.constraints = constraints
        self.optimizer = torch.optim.Adam([self.cont6d], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()

        glb = self.forward(self.cont6d, self.pos, self.parents)
        loss = self.crit(glb, self.constraints)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()
    
    def forward(self, cont6d, pos, parents):
        
        # positions = self.skeleton.forward_kinematics_real_cont6d(cont6d_params, r_pos, r_rot_quat, self.offset)
        quats = cont6d2q(cont6d)
        _, positions = quat_fk(quats, pos, parents,)
        return positions
