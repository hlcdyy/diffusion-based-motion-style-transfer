import numpy 
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from data_loaders.humanml.common.rotation import rotm2axangle, rotm2q, q2axangle
from fractions import Fraction
from data_loaders.humanml.common.quaternion import qslerp, lerp
import imageio
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from data_loaders.humanml.common.bvh_utils import process_file, recover_from_ric
import pickle as pk
from data_loaders.humanml.common.skeleton import Skeleton
import os
from data_loaders.humanml.scripts.motion_process import uniform_skeleton_basic


male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
faces = c2c(male_bm.f)

female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

'''Get offsets of target skeleton'''
data_dir = "./processed_data/HumanML3D/new_joints"
example_id = '001234'
example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
tgt_skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, 'cpu')
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])


# trans_matrix = np.array([[1.0, 0.0, 0.0],
#                             [0.0, 0.0, 1.0],
#                             [0.0, 1.0, 0.0]])
trans_matrix = np.array([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])
ex_fps = 20

def lcm_multiple(numbers):
    return np.lcm.reduce(numbers)

def downsample(rotations, positions, downsample_rate):
    fraction = Fraction(downsample_rate)
    lcm = lcm_multiple([fraction.numerator, fraction.denominator])
    upsample_rate = int(lcm / fraction.numerator)
    new_downsample_rate = int(lcm / fraction.denominator)
    t = torch.from_numpy(np.linspace(0, 1, upsample_rate+1))[:-1]
    t = t.to(rotations.device)

    new_rotations = qslerp(rotations[0:-1, ...], rotations[1:, ...], t)
    new_rotations = new_rotations.permute(1, 0, 2, 3).reshape((-1,) + tuple(new_rotations.shape[2:]))
    new_positions = lerp(positions[0:-1, ...], positions[1:, ...], t)
    new_positions = new_positions.permute(1, 0 ,2).reshape((-1,) + tuple(new_positions.shape[2:]))

    rotations_tmp = new_rotations[0::new_downsample_rate, ...]
    positions_tmp = new_positions[0::new_downsample_rate, ...]

    return rotations_tmp, positions_tmp

def joints_downsample(joints, downsample_rate):
    fraction = Fraction(downsample_rate)
    lcm = lcm_multiple([fraction.numerator, fraction.denominator])
    upsample_rate = int(lcm / fraction.numerator)
    new_downsample_rate = int(lcm / fraction.denominator)
    t = torch.from_numpy(np.linspace(0, 1, upsample_rate+1))[:-1]
    t = t.to(joints.device)

    new_joints = lerp(joints[0:-1, ...], joints[1:, ...], t)
    new_joints = new_joints.permute(1, 0 ,2, 3).reshape((-1,) + tuple(new_joints.shape[2:]))

    new_joints = new_joints[0::new_downsample_rate, ...]
    
    return new_joints

def amass_to_pose(src_path, fps=25, trans_path="", with_trans=False):
    if src_path[-2:] == 'pt':
        bdata = torch.load(src_path)
        bdata = bdata[0]
        theta_mats = bdata["pred_theta_mats"].reshape(-1, 24, 3, 3)
        betas = bdata["pred_shape"].mean(0).unsqueeze(0)       
        joints = bdata["pred_xyz_jts_24_struct"].reshape(-1, 24, 3)
        transl = bdata["transl"]
    
    elif src_path[-2:] == 'pk':
        with open(src_path, 'rb') as f:
            bdata = pk.load(f)
        theta_mats = torch.from_numpy(bdata["pred_thetas"].reshape(-1, 24, 3, 3))
        betas = torch.from_numpy(bdata["pred_betas"]).mean(0).unsqueeze(0)
        joints = bdata["pred_xyz_24_struct"].reshape(-1, 24, 3)
        joints = torch.from_numpy(joints)
        transl = bdata["transl"]
        transl = torch.from_numpy(transl) 
    
    if src_path[-3:] == 'pkl':
        with open(src_path, 'rb') as f:
            bdata = pk.load(f)
        bdata = bdata[0]
        theta_mats = torch.from_numpy(bdata["smpl_pose_quat_wroot"])
        betas = torch.from_numpy(bdata["smpl_beta"]).mean(0).unsqueeze(0)
        transl = torch.from_numpy(bdata["root_trans"])
    
    if trans_path:
        with open(trans_path, 'rb') as f:
            trans_data = pk.load(f)
        trans_data = trans_data[0]
        transl = torch.from_numpy(trans_data["root_trans"])
        
        if transl.shape[0] < joints.shape[0]:
            joints = joints[:transl.shape[0]]
            theta_mats = theta_mats[:transl.shape[0]]
        else:
            transl = transl[:joints.shape[0]]

    if theta_mats.dim() > 3:
        theta_quats = rotm2q(theta_mats)
    else:
        theta_quats = theta_mats
    # joints = bdata["pred_xyz_jts_24"].reshape(-1, 24, 3)
   
    
    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    bm = male_bm
    
    down_sample = (fps / ex_fps)
    theta_quats, transl = downsample(theta_quats, transl, down_sample)

    frame_number = theta_quats.shape[0]
#     print(frame_number)
#     print(fps)
    

    poses = q2axangle(theta_quats).reshape(-1, 24 * 3) # B 24 3
    with torch.no_grad():
        for fId in range(0, frame_number):
            root_orient = poses[fId:fId+1, :3].to(comp_device)
            pose_body = poses[fId:fId+1, 3:66].to(comp_device)
            betas = betas.to(comp_device)
            trans = transl[fId:fId+1].to(comp_device)
            body = bm(pose_body=pose_body, betas=betas, root_orient=root_orient)
            if with_trans:
                joint_loc = body.Jtr[0] + trans
            else:
                joint_loc = body.Jtr[0]
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)
    
    pose_seq_np = pose_seq.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    pose_seq_np_n[..., 1] *= -1 
    pose_seq_np_n = pose_seq_np_n[..., :22, :]
    
    try:
        joints = joints_downsample(joints, down_sample)
        joints = joints.detach().cpu().numpy()
        joints = np.dot(joints, trans_matrix)
        joints[..., 1] *= -1
        joints = joints[..., :22, :]
        new_transl = transl[:, [2, 1, 0]].cpu().numpy()
        new_transl[..., 1] *= -1
        if with_trans:
            joints += new_transl[:, np.newaxis]
    except:
        joints = pose_seq_np_n
    return pose_seq_np_n, joints


def pos2hmlrep(joints):
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    # retargeting
    l_idx = [5, 8]
    joints = uniform_skeleton_basic(joints, tgt_offsets, n_raw_offsets, t2m_kinematic_chain, l_idx, face_joint_indx)
    data, global_positions, positions, l_velocity = process_file(joints, face_joint_indx, fid_l, fid_r, 0.002, n_raw_offsets, t2m_kinematic_chain)
    return data

    
