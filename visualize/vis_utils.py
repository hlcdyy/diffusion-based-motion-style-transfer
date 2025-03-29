from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
from data_loaders.humanml.common.rotation import cont6d2q, wrap
from data_loaders.humanml.common.bvh_utils import Anim,save_bvh, Butterworth

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        self.vertices += self.root_loc

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)

def joints2rotation(joints, num_smplify_iters=150):
    frames, njoints, nfeats = joints.shape
    MINS = joints.min(axis=0).min(axis=0)

    height_offset = MINS[1]
    joints[:, :, 1] -= height_offset
    
    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True, num_smplify_iters=num_smplify_iters)
    motion_tensor, opt_dict = j2s.joint2smpl(joints)  # [nframes, njoints, 3]
    
    return motion_tensor

def joints2bvh(path, joints, real_offset, kinematic_chain, names=None, num_smplify_iters=150, Butterworth_all=False):

    motion_tensor = joints2rotation(joints, num_smplify_iters) # [1, 25, 6, seq]
    motion_tensor = motion_tensor.squeeze(0).permute(2, 0, 1) # Seq J 6

    # head = 15
    # neck = 12
    joint_indices = [12, 15]

    if Butterworth_all:
        joint_indices = range(0, motion_tensor.shape[1])
    motion_tensor = motion_tensor.detach().cpu().numpy()

    for joint in joint_indices:
        for j in range(motion_tensor.shape[-1]):
            motion_tensor[..., joint, j] = Butterworth(motion_tensor[..., joint, j], 1/20, 1.8)

    new_quats = wrap(cont6d2q, motion_tensor[:, :22, :]) # Seq J 4

    real_offset = real_offset.copy()

    parents = [-1] * real_offset.shape[0]  

    for idx, chain in enumerate(kinematic_chain):
        for index, i in enumerate(chain[1:]):
            parents[i] = chain[index]       
    
    real_offset[0, :] = np.zeros((1, 3), np.float32)
    
    new_pos = real_offset[None, ...].repeat(new_quats.shape[0], axis=0)
    new_pos[:, 0, :] = motion_tensor[:, -1, :3]
    # offset, pos, quats, parents, names
    anim = Anim(new_quats, new_pos, real_offset, parents, names)

    save_bvh(path, anim, 1/20)

    