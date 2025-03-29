import numpy as np

HML_JOINT_NAMES = [
    'Hips',
    'Spine',
    'Chest',
    'Neck',
    'Head',
    'Shoulder_L',
    'UpperArm_L',
    'LowerArm_L',
    'Hand_L',
    'Shoulder_R',
    'UpperArm_R',
    'LowerArm_R',
    'Hand_R',
    'UpperLeg_L',
    'LowerLeg_L',
    'Foot_L',
    'Toes_L',
    'UpperLeg_R',
    'LowerLeg_R',
    'Foot_R',
    'Toes_R',
]

BVH_JOINT_NAMES = [
    'Hips',
    'Spine',
    'Chest',
    'Neck',
    'Head',
    'Shoulder_L',
    'UpperArm_L',
    'LowerArm_L',
    'Hand_L',
    'Shoulder_R',
    'UpperArm_R',
    'LowerArm_R',
    'Hand_R',
    'UpperLeg_L',
    'LowerLeg_L',
    'Foot_L',
    'Toes_L',
    'UpperLeg_R',
    'LowerLeg_R',
    'Foot_R',
    'Toes_R',
]


NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 21 body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['Hips', 'UpperLeg_L','LowerLeg_L','Foot_L','Toes_L','UpperLeg_R','LowerLeg_R','Foot_R','Toes_R']]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY.repeat(6)))

HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(6))))

HML_NONE_MASK = np.concatenate(([False]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(6))))

HML_YROTATION_MASK = np.concatenate(([True] + [False] * 3,
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(6))))


HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(6),
                                     ))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK
HML_TRAJ_MASK = np.zeros_like(HML_ROOT_MASK)
HML_TRAJ_MASK[1:3] = True

NUM_HML_FEATS = 190


def expand_mask(mask, shape):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_feat, 1, seq_len))
    """
    _, num_feat, _, _ = shape
    return np.ones(shape) * mask.reshape((1, num_feat, 1, -1))

def get_joints_mask(join_names):
    joins_mask = np.array([joint_name in join_names for joint_name in HML_JOINT_NAMES])
    mask = np.concatenate(([False]*(1+2+1),
                                joins_mask[1:].repeat(3),
                                np.zeros_like(joins_mask.repeat(6)),
                                ))
    return mask

def get_batch_joint_mask(shape, joint_names):
    return expand_mask(get_joints_mask(joint_names), shape)

def get_in_between_mask(shape, lengths, prefix_end, suffix_end):
    mask = np.ones(shape)  # True means use gt motion
    for i, length in enumerate(lengths):
        start_idx, end_idx = int(prefix_end * length), int(suffix_end * length)
        mask[i, :, :, start_idx: end_idx] = 0  # do inpainting in those frames
    return mask

def get_prefix_mask(shape, prefix_length=20):
    _, num_feat, _, seq_len = shape
    prefix_mask = np.concatenate((np.ones((num_feat, prefix_length)), np.zeros((num_feat, seq_len - prefix_length))), axis=-1)
    return expand_mask(prefix_mask, shape)

def get_inpainting_mask(mask_name, shape, **kwargs):
    mask_names = mask_name.split(',')
    
    mask = np.zeros(shape)
    if 'in_between' in mask_names:
        mask = np.maximum(mask, get_in_between_mask(shape, **kwargs))
    
    if 'none' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_NONE_MASK, shape))
        
    if 'root' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_MASK, shape))
    
    if 'root_horizontal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))

    if 'y_rotation' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_YROTATION_MASK, shape))

    if 'prefix' in mask_names:
        mask = np.maximum(mask, get_prefix_mask(shape, **kwargs))

    if 'upper_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_UPPER_BODY_MASK, shape))
    
    if 'lower_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_LOWER_BODY_MASK, shape))
    
    return np.maximum(mask, get_batch_joint_mask(shape, mask_names))