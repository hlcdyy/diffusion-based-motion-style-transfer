import numpy as np
import re, os, ntpath
from copy import deepcopy
import torch.cuda
from utils.rotation import *
from data_loaders.humanml.common.skeleton import *
from fractions import Fraction
from data_loaders.humanml.common.Kinematics import InverseKinematics_hmlvec, InverseKinematics_quats
from tqdm import tqdm

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones, end_offsets=None):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        if bones is None:
            bones = ["joint_" + str(i) for i in range(len(parents))]
            self.bones = bones
        else:
            self.bones = bones
        self.not_endsite = []
        self.endsite = []
        self.end_offsets = end_offsets
        for i in range(len(bones)):
            if bones[i] != 'End Site':
                self.not_endsite.append(i)
            else:
                self.endsite.append(i)
        # self.end_offsets = self.offsets[self.endsite]

        self.notend_offsets = self.offsets[self.not_endsite]
        if len(self.endsite) != 0:
            self.format_end = True
            self.simple_parents = self.parents[self.not_endsite]
            for i, index in enumerate(self.simple_parents[1:]):
                self.simple_parents[i+1] = self.not_endsite.index(index)

            self.simple_bones = []
            for name in self.bones:
                if name != 'End Site':
                    self.simple_bones.append(name)
        else:
            self.format_end = False
            self.simple_parents = self.parents
            self.simple_bones = self.bones

    @property
    def shape(self): return (self.quats.shape[0], self.quats.shape[1])

    def clip(self, slice):
        self.quats = self.quats[slice]
        self.pos = self.pos[slice]


def read_bvh(filename, start=None, end=None, order=None, downsample_rate=None, end_sites=False):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1

    not_end_index = []
    end_index = []
    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\S+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "{" in line: continue

        if "}" in line:
            active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\S+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "End Site" in line:
            names.append('End Site')
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            end_index.append(active)
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = int(end - start)
            elif start and end is None:
                fnum = int(fmatch.group(1)) - int(start)
            elif end and start is None:
                fnum = int(end)
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations  = np.zeros((fnum, len(orients), 3), dtype=np.float32)
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if start and i < start:
            i += 1
            continue
        if end and i >= end:
            i += 1
            continue

        # if (start and end) and (i < start or i >= end):
        #     i += 1
        #     continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - int(start) if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, not_end_index] = data_block[3:].reshape(len(not_end_index), 3)
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N-len(not_end_index), axis=0)
            elif channels == 6:
                data_block = data_block.reshape(len(not_end_index), 6)
                positions[fi, not_end_index] = data_block[:, 0:3]
                rotations[fi, not_end_index] = data_block[:, 3:6]
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N - len(not_end_index), axis=0)
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(len(not_end_index) - 1, 9)
                rotations[fi, not_end_index[1:]] = data_block[:, 3:6]
                positions[fi, not_end_index[1:]] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()
    positions = positions.astype(np.float32)
    offsets = offsets.astype(np.float32)
    # print(rotations.dtype, positions.dtype, offsets.dtype)
    rotations = wrap(eul2q, np.radians(rotations), order)
    rotations = wrap(remove_quat_discontinuities, rotations)

    # parents with not endsites:
    simple_parents = parents[not_end_index]
    for i, index in enumerate(simple_parents[1:]):
        simple_parents[i+1] = not_end_index.index(index)
    simple_names = []
    for i, name in enumerate(names):
        if i in not_end_index:
            simple_names.append(name)


    if downsample_rate is not None:
        Anim_list = []
        if round(downsample_rate) == downsample_rate:
            for i in range(downsample_rate):
                rotations_tmp = rotations[i::downsample_rate, ...]
                positions_tmp = positions[i::downsample_rate, ...]
                if end_sites:
                    Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
                else:
                    Anim_list.append(Anim(rotations_tmp[:, not_end_index, :],
                                        positions_tmp[:, not_end_index, :],
                                        offsets[not_end_index, :], simple_parents, simple_names))
            return Anim_list
        else:
            # first upsample and then downsmaple

            def decimal_to_fraction(decimal_number):
                # 转换成整数部分和小数部分
                whole_number, digits = str(decimal_number).split('.')
                numerator = int(digits)
                denominator = 10 ** len(digits)
                # 合并整数部分
                numerator += int(whole_number) * denominator
                # 简化分数
                fraction = Fraction(numerator, denominator).limit_denominator()
                return fraction
            
            def lcm_multiple(numbers):
                return np.lcm.reduce(numbers)
            # fraction = Fraction(downsample_rate)
            fraction = decimal_to_fraction(downsample_rate)
            lcm = lcm_multiple([fraction.numerator, fraction.denominator])
            upsample_rate = int(lcm / fraction.numerator)
            new_downsample_rate = int(lcm / fraction.denominator)
            t = torch.from_numpy(np.linspace(0, 1, upsample_rate+1))[:-1]
            new_rotations = wrap(qslerp, rotations[0:-1, ...], rotations[1:, ...], t)
            new_rotations = new_rotations.transpose(1, 0, 2, 3).reshape((-1,) + tuple(new_rotations.shape[2:]))
            new_positions = wrap(lerp, positions[0:-1, ...], positions[1:, ...], t)
            new_positions = new_positions.transpose(1, 0, 2, 3).reshape((-1,) + tuple(new_positions.shape[2:]))
            rotations_tmp = new_rotations[0::new_downsample_rate, ...]
            positions_tmp = new_positions[0::new_downsample_rate, ...]
            rotations_tmp = rotations_tmp.astype(np.float32)
            positions_tmp = positions_tmp.astype(np.float32)
            if end_sites:
                Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
            else:
                Anim_list.append(Anim(rotations_tmp[:, not_end_index, :],
                                    positions_tmp[:, not_end_index, :],
                                    offsets[not_end_index, :], simple_parents, simple_names))
            return Anim_list

            
    else:
        if end_sites:
            return Anim(rotations, positions, offsets, parents, names, offsets[end_index, :])
        else:
            return Anim(rotations[:, not_end_index, :], positions[:, not_end_index, :],
                        offsets[not_end_index, :], simple_parents, simple_names, offsets[end_index, :])
        
def read_bvh_raw_motion(filename, start=None, end=None, order=None, downsample_rate=None, end_sites=False):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1

    not_end_index = []
    end_index = []
    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\S+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "{" in line: continue

        if "}" in line:
            active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\S+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "End Site" in line:
            names.append('End Site')
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            end_index.append(active)
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = int(end - start)
            elif start and end is None:
                fnum = int(fmatch.group(1)) - int(start)
            elif end and start is None:
                fnum = int(end)
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations  = np.zeros((fnum, len(orients), 3), dtype=np.float32)
            raw_motions = []
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if start and i < start:
            i += 1
            continue
        if end and i >= end:
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            raw_motions.append(data_block)
            N = len(parents)
            fi = i - int(start) if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, not_end_index] = data_block[3:].reshape(len(not_end_index), 3)
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N-len(not_end_index), axis=0)
            elif channels == 6:
                data_block = data_block.reshape(len(not_end_index), 6)
                positions[fi, not_end_index] = data_block[:, 0:3]
                rotations[fi, not_end_index] = data_block[:, 3:6]
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N - len(not_end_index), axis=0)
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(len(not_end_index) - 1, 9)
                rotations[fi, not_end_index[1:]] = data_block[:, 3:6]
                positions[fi, not_end_index[1:]] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()
    raw_motions = np.stack(raw_motions, 0)

    positions = positions.astype(np.float32)
    offsets = offsets.astype(np.float32)
    # print(rotations.dtype, positions.dtype, offsets.dtype)
    rotations = wrap(eul2q, np.radians(rotations), order)
    rotations = wrap(remove_quat_discontinuities, rotations)

    # parents with not endsites:
    simple_parents = parents[not_end_index]
    for i, index in enumerate(simple_parents[1:]):
        simple_parents[i+1] = not_end_index.index(index)
    simple_names = []
    for i, name in enumerate(names):
        if i in not_end_index:
            simple_names.append(name)
        

    if downsample_rate is not None:
        Anim_list = []
        euler_list = []
        if round(downsample_rate) == downsample_rate:
            for i in range(downsample_rate):
                rotations_tmp = rotations[i::downsample_rate, ...]
                positions_tmp = positions[i::downsample_rate, ...]
                euler_list.append(raw_motions[i::downsample_rate, ...])
                if end_sites:
                    Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
                else:
                    Anim_list.append(Anim(rotations_tmp[:, not_end_index, :],
                                        positions_tmp[:, not_end_index, :],
                                        offsets[not_end_index, :], simple_parents, simple_names))
            return Anim_list, euler_list
        else:
            # first upsample and then downsmaple
            def lcm_multiple(numbers):
                return np.lcm.reduce(numbers)
            fraction = Fraction(downsample_rate)
            lcm = lcm_multiple([fraction.numerator, fraction.denominator])
            upsample_rate = int(lcm / fraction.numerator)
            new_downsample_rate = int(lcm / fraction.denominator)
            t = torch.from_numpy(np.linspace(0, 1, upsample_rate+1))[:-1]
            new_rotations = wrap(qslerp, rotations[0:-1, ...], rotations[1:, ...], t)
            new_rotations = new_rotations.transpose(1, 0, 2, 3).reshape((-1,) + tuple(new_rotations.shape[2:]))
            new_positions = wrap(lerp, positions[0:-1, ...], positions[1:, ...], t)
            new_positions = new_positions.transpose(1, 0, 2, 3).reshape((-1,) + tuple(new_positions.shape[2:]))
            rotations_tmp = new_rotations[0::new_downsample_rate, ...]
            positions_tmp = new_positions[0::new_downsample_rate, ...]
            rotations_tmp = rotations_tmp.astype(np.float32)
            positions_tmp = positions_tmp.astype(np.float32)
            if end_sites:
                Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
            else:
                Anim_list.append(Anim(rotations_tmp[:, not_end_index, :],
                                    positions_tmp[:, not_end_index, :],
                                    offsets[not_end_index, :], simple_parents, simple_names))
            return Anim_list
      
    else:
        if end_sites:
            return Anim(rotations, positions, offsets, parents, names, offsets[end_index, :]), raw_motions
        else:
            return Anim(rotations[:, not_end_index, :], positions[:, not_end_index, :],
                        offsets[not_end_index, :], simple_parents, simple_names, offsets[end_index, :]), raw_motions


def save_bvh(filename, anim, frametime=1.0 / 24.0, order='zyx',
             positions=False, orients=True, end_offset=None):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    names = anim.bones
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]


    if anim.quats.shape[1] < anim.offsets.shape[0]:
       # vailid quats, offsets include endsites
        with_end = False
        end_offset = anim.end_offsets
        anim.offsets = anim.notend_offsets
        names = anim.simple_bones
        anim.parents = anim.simple_parents
        not_end_index = None

    elif 'End Site' in names:
        with_end = True
        end_offset = None
        not_end_index = anim.not_endsite
    else:
        with_end = False
        if end_offset == None:
            end_offset = anim.end_offsets
        else:
            end_offset = end_offset
        not_end_index = None


    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_joint_seq = [0]
        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t, end_offset = save_joint(f, anim, names, t, i, save_joint_seq, order=order,
                               positions=positions, with_end=with_end, end_offset=end_offset)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        # if orients:
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        # else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        anim.quats = anim.quats[:, save_joint_seq, :]

        if not_end_index is not None:
            rots = np.degrees(wrap(q2eul, anim.quats[:, not_end_index, :], order[::-1]))
        else:
            rots = np.degrees(wrap(q2eul, anim.quats, order[::-1]))
        poss = anim.pos

        if not_end_index is not None:
            rot_jnum = len(not_end_index)
        else:
            rot_jnum = anim.shape[1]

        for i in range(anim.shape[0]):
            for j in range(rot_jnum):

                if positions or j == 0:
                    f.write("%f %f %f %f %f %f " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

                else:

                    f.write("%f %f %f " % (
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

            f.write("\n")


def save_bvh_raw_motion(filename, anim, raw_motions, frametime=1.0 / 24.0, order='zyx',
             positions=False, orients=True, end_offset=None):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    names = anim.bones
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]


    if anim.quats.shape[1] < anim.offsets.shape[0]:
       # vailid quats, offsets include endsites
        with_end = False
        end_offset = anim.end_offsets
        anim.offsets = anim.notend_offsets
        names = anim.simple_bones
        anim.parents = anim.simple_parents
        not_end_index = None

    elif 'End Site' in names:
        with_end = True
        end_offset = None
        not_end_index = anim.not_endsite
    else:
        with_end = False
        if end_offset == None:
            end_offset = anim.end_offsets
        else:
            end_offset = end_offset
        not_end_index = None


    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_joint_seq = [0]
        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t, end_offset = save_joint(f, anim, names, t, i, save_joint_seq, order=order,
                               positions=positions, with_end=with_end, end_offset=end_offset)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % raw_motions.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        # if orients:
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        # else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        # anim.quats = anim.quats[:, save_joint_seq, :]

        # if not_end_index is not None:
        #     rots = np.degrees(wrap(q2eul, anim.quats[:, not_end_index, :], order[::-1]))
        # else:
        #     rots = np.degrees(wrap(q2eul, anim.quats, order[::-1]))
        # poss = anim.pos

        if not_end_index is not None:
            rot_jnum = len(not_end_index)
        else:
            rot_jnum = anim.shape[1]

        for i in range(raw_motions.shape[0]):
            f.write(" ".join(map(str, raw_motions[i])))
            # for j in range(rot_jnum):

            #     if positions or j == 0:
            #         f.write("%f %f %f %f %f %f " % (
            #             # poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
            #             # rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))
                        
            #     else:

            #         f.write("%f %f %f " % (
            #             rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

            f.write("\n")

def save_joint(f, anim, names, t, i, save_joint_seq, order='zyx', positions=False, with_end=False, end_offset=None):
    save_joint_seq.append(i)
    if with_end:
        if names[i] == 'End Site':
            f.write("%s%s\n" % (t, names[i]))
        else:
            f.write("%sJOINT %s\n" % (t, names[i]))
    else:
        f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))

    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                            channelmap_inv[order[0]],
                                                                            channelmap_inv[order[1]],
                                                                            channelmap_inv[order[2]]))
    else:
        if with_end:
            if names[i] != 'End Site':
                f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                                     channelmap_inv[order[0]], channelmap_inv[order[1]],
                                                     channelmap_inv[order[2]]))
        else:
            f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                             channelmap_inv[order[0]], channelmap_inv[order[1]],
                                             channelmap_inv[order[2]]))

    end_site = True

    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t, end_offset = save_joint(f, anim, names, t, j, save_joint_seq, order=order,
                                       positions=positions, with_end=with_end, end_offset=end_offset)
            end_site = False

    if not with_end:
        if end_site:
            f.write("%sEnd Site\n" % t)
            f.write("%s{\n" % t)
            t += '\t'
            if end_offset is None:
                f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
            else:
                f.write("%sOFFSET %f %f %f\n" % (t, end_offset[0, 0], end_offset[0, 1], end_offset[0, 2]))
                end_offset = end_offset[1:, :]
            t = t[:-1]
            f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)

    return t, end_offset


def skelchains2parents(skel_chains):
    a = set()
    for limbs in skel_chains:
        a = a | set(limbs)
    parents = [-1] * len(a)
    for limbs in skel_chains:
        for i, joint in enumerate(limbs):
            if i == 0:
                continue
            parents[joint] = limbs[i-1]
    return parents


def compute_height(offsets, ind_lf, ind_ln):
    """
    ind_lf: left_foot index
    ind_ln: left_knee index
    """
    # compute human height by left leg
    lower_leg_len = np.linalg.norm(offsets[ind_lf, :], axis=-1, ord=2)
    upper_leg_len = np.linalg.norm(offsets[ind_ln, :], axis=-1, ord=2)
    return lower_leg_len + upper_leg_len

def extract_chains(anim, limbs=['RightFoot', 'LeftFoot', 'Head', 'RightHand', 'LeftHand']):
    parents = anim.parents
    degree = [0] * 300
    seq_list = []

    limb_indices = []
    for name in limbs:
        limb_indices.append(anim.bones.index(name))

    for i, p in enumerate(parents):
        degree[i] += 1
        if p != -1:
            degree[p] += 1

    def find_seq(j, seq):
        nonlocal degree, parents, seq_list

        if degree[j] > 2 and j > 1:
            seq_list.append(seq)
            seq = []

        if degree[j] == 1:
            seq_list.append(seq + [j])
            return

        for idx, p in enumerate(parents):
            if p == j:
                find_seq(idx, seq + [j])
    find_seq(0, [])
    seq_list_new = []
    try:
        head_idx = limbs.index('Head')
    except:
        head_idx = limbs.index('head')

    def combine_former_seq(seq_list, seq):
        for _seq in seq_list:
            if _seq[-1] == seq[0] - 1:
                new_seq = _seq + seq
        return new_seq

    for i, idx in enumerate(limb_indices):
        for seq in seq_list:
            if idx in seq:
                if i != head_idx:
                    seq_list_new.append(seq)
                else:
                    head_seq = combine_former_seq(seq_list, seq)
                    seq_list_new.append(head_seq)
    raw_offsets = np.where(abs(anim.offsets) < 0.001, 0, anim.offsets)
    real_offsets = raw_offsets.copy()
    raw_offsets = raw_offsets / (np.linalg.norm(raw_offsets, ord=2, axis=-1, keepdims=True) + 1e-9)
    raw_offsets[0, :] = [0., 0., 0.]
    return seq_list_new, raw_offsets, real_offsets


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints



def process_file(positions, face_joint_indx, fid_l, fid_r, feet_thre, n_raw_offsets, kinematic_chain):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]


    # '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0., 0., 1.]])

    root_quat_init = wrap(quatbetween, forward_init, target)  # (1, 4)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  # Seq J 4

    positions_b = positions.copy()

    positions = wrap(qrot, root_quat_init, positions)


    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """
    # fid_l = [3, 4]
    # fid_r = [7, 8]

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
        
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''

        positions = wrap(qrot, np.repeat(wrap(qinv, r_rot)[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = remove_quat_discontinuities(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = wrap(qrot, qinv(r_rot[1:]), velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = wrap(qmultipy, r_rot[1:], wrap(qinv, r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = wrap(q2cont6d, quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        # print(r_rot[1])
        # exit()
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = wrap(qrot, wrap(qinv, r_rot[1:]), velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = wrap(qmultipy, r_rot[1:], wrap(qinv, r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    # Revised by HL  
    local_vel = wrap(qrot, wrap(qinv, np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1)),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data # 4
    data = np.concatenate([data, ric_data[:-1]], axis=-1) # 3 * (jnum-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1) # 6 * (jnum-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)  # 3 * jnum
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # 4

    return data, global_positions, positions, l_velocity


def process_file_with_rotation(positions, rotations, face_joint_indx, fid_l, fid_r, feet_thre, n_raw_offsets, kinematic_chain):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]


    # '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0., 0., 1.]])

    root_quat_init = wrap(quatbetween, forward_init, target)  # (1, 4)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  # Seq J 4

    positions_b = positions.copy()

    positions = wrap(qrot, root_quat_init, positions)
    
    rotations[:, 0] = qmul_np(root_quat_init[:, 0, :], rotations[:, 0])


    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """
    # fid_l = [3, 4]
    # fid_r = [7, 8]

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
        
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions, rotations):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''

        positions = wrap(qrot, np.repeat(wrap(qinv, r_rot)[:, None], positions.shape[1], axis=1), positions)
        rotations[:, 0, :] = qmul_np(qinv_np(r_rot), rotations[:, 0, :])
        return positions, rotations

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = remove_quat_discontinuities(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = wrap(qrot, qinv(r_rot[1:]), velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = wrap(qmultipy, r_rot[1:], wrap(qinv, r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        quat_params = quat_params.astype(np.float32)
        '''Quaternion to continuous 6D'''
        cont_6d_params = wrap(q2cont6d, quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        # print(r_rot[1])
        # exit()
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = wrap(qrot, wrap(qinv, r_rot[1:]), velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = wrap(qmultipy, r_rot[1:], wrap(qinv, r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions, rotations = get_rifke(positions, rotations)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    # rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    rot_data = quaternion_to_cont6d_np(rotations).reshape(len(rotations), -1)


    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    # Revised by HL  
    local_vel = wrap(qrot, wrap(qinv, np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1)),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data # 4
    data = np.concatenate([data, ric_data[:-1]], axis=-1) # 3 * (jnum-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1) # 6 * (jnum-1)
    #     print(data.shape, local_vel.shape)
    # data = np.concatenate([data, local_vel], axis=-1)  # 3 * jnum
    # data = np.concatenate([data, feet_l, feet_r], axis=-1) # 4

    return data, global_positions, positions, l_velocity



# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
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


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = q2cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_from_real_rot(data, joints_num, skeleton):
    # B Seq d
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    cont6d_params = data[..., 4 + (joints_num-1) * 3:].reshape(data.shape[:-1] + (joints_num, 6))

    positions = skeleton.forward_kinematics_real_cont6d(cont6d_params, r_pos, r_rot_quat, skeleton._offset)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(r_rot_quat[..., None, :].expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def process_pos_from_real_rot(data, joints_num, skel):
    # S d 
    r_rot_quat, _ = recover_root_rot_pos(data)
    
    pos = recover_from_real_rot(data, joints_num, skel)
    pos[..., 0] -= pos[..., 0:1, 0]
    pos[..., 2] -= pos[..., 0:1, 2]

    pos = pos[..., 1:, :]
    pos = qrot(qinv(r_rot_quat)[..., None, :], pos)
    pos = pos.reshape(pos.shape[:-2] + (-1,))

    return pos



def output_bvh(path, data, joints_num, kinematic_chain, tgt_offsets):
    device = data.device
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # joint 6d rotation
    cont6d = data[..., 4 + (joints_num - 1) * 3: 4+(joints_num - 1) * 9].view(data.shape[:-1] + (-1, 6))
    quats = cont6d2q(cont6d)  # (*, J-1, 4)
    world_quats = torch.zeros_like(quats)
    for chain in kinematic_chain:
        R = r_rot_quat
        for j in chain[1:]:
            R = qmultipy(R, quats[..., j-1, :])
            world_quats[..., j-1, :] = R
    world_quats = torch.cat((r_rot_quat[:, None], world_quats), dim=-2)

    new_offsets = tgt_offsets.clone()
    new_kinematic_chain = deepcopy(kinematic_chain)
    for i, chain in enumerate(new_kinematic_chain):
        now_joint = chain[1]
        for chain2 in new_kinematic_chain:
            for index, j in enumerate(chain2):
                if j >= now_joint:
                    chain2[index] += 1
        chain.insert(1, now_joint)

    insert_list = []
    for chain in kinematic_chain:
        insert_list.append(chain[1])
    insert_list = sorted(insert_list, reverse=True)

    zero = torch.zeros((1, 3)).to(tgt_offsets.device)
    for index in insert_list:
        new_offsets = torch.cat((new_offsets[:index, :], zero, new_offsets[index:, :]), dim=0)

    new_world_quats = torch.zeros((world_quats.shape[:-2] + (new_offsets.shape[0], 4))).to(device)
    new_world_quats[..., 0] = 1.

    new_parents = [-1] * new_offsets.shape[0]
    for idx, chain in enumerate(new_kinematic_chain):
        new_world_quats[..., chain[0], :] = world_quats[..., kinematic_chain[idx][0], :]
        for index, i in enumerate(chain[1:]):
            new_parents[i] = chain[index]
            if index != len(chain[1:]) -1:
                new_world_quats[..., i, :] = world_quats[..., kinematic_chain[idx][index+1], :]
            else:
                new_world_quats[..., i, :] = world_quats[..., kinematic_chain[idx][index], :]
                

    # world rotations -> local rotations
    new_quats = torch.cat(
        [new_world_quats[..., :1, :],
        qmultipy(qinv(new_world_quats[..., new_parents[1:], :]), new_world_quats[..., 1:, :]),
        ],dim=-2)
    
    new_pos = new_offsets.expand(r_pos.shape[0], -1, -1).clone()
    new_pos[:, 0, :] = r_pos
    # offset, pos, quats, parents, names
    anim = Anim(new_quats, new_pos, new_offsets, new_parents, None)

    save_bvh(path, anim, 1/20)


def output_bvh_with_pos(path, data, joints_num, kinematic_chain, tgt_offsets, n_raw_offsets, face_joint_indx, bone_names=None):
    device = data.device

    positions = recover_from_ric(data, joints_num)

    r_rot_quat, r_pos = recover_root_rot_pos(data)

    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    
    positions = positions.cpu().numpy()
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    
    quats = torch.Tensor(quat_params).to(device)  
    quats = quats[..., 1:, :] # (*, J-1, 4)
    world_quats = torch.zeros_like(quats)
    for chain in kinematic_chain:
        R = r_rot_quat
        for j in chain[1:]:
            R = qmultipy(R, quats[..., j-1, :])
            world_quats[..., j-1, :] = R
    world_quats = torch.cat((r_rot_quat[:, None], world_quats), dim=-2)

    new_offsets = tgt_offsets.clone()
    new_kinematic_chain = deepcopy(kinematic_chain)
    for i, chain in enumerate(new_kinematic_chain):
        now_joint = chain[1]
        for chain2 in new_kinematic_chain:
            for index, j in enumerate(chain2):
                if j >= now_joint:
                    chain2[index] += 1
        chain.insert(1, now_joint)

    insert_list = []
    for chain in kinematic_chain:
        insert_list.append(chain[1])
    insert_list = sorted(insert_list, reverse=True)

    zero = torch.zeros((1, 3)).to(tgt_offsets.device)
    for index in insert_list:
        new_offsets = torch.cat((new_offsets[:index, :], zero, new_offsets[index:, :]), dim=0)

    new_world_quats = torch.zeros((world_quats.shape[:-2] + (new_offsets.shape[0], 4))).to(device)
    new_world_quats[..., 0] = 1.

    new_parents = [-1] * new_offsets.shape[0]
    for idx, chain in enumerate(new_kinematic_chain):
        new_world_quats[..., chain[0], :] = world_quats[..., kinematic_chain[idx][0], :]
        for index, i in enumerate(chain[1:]):
            new_parents[i] = chain[index]
            if index != len(chain[1:]) -1:
                new_world_quats[..., i, :] = world_quats[..., kinematic_chain[idx][index+1], :]
            else:
                new_world_quats[..., i, :] = world_quats[..., kinematic_chain[idx][index], :]
                

    # world rotations -> local rotations
    new_quats = torch.cat(
        [new_world_quats[..., :1, :],
        qmultipy(qinv(new_world_quats[..., new_parents[1:], :]), new_world_quats[..., 1:, :]),
        ],dim=-2)
    
    new_pos = new_offsets.expand(r_pos.shape[0], -1, -1).clone()
    new_pos[:, 0, :] = r_pos
    # offset, pos, quats, parents, names
    anim = Anim(new_quats, new_pos, new_offsets, new_parents, bone_names)

    save_bvh(path, anim, 1/20)


def output_bvh_with_22rot(path, pos_data, data, joints_num, kinematic_chain, tgt_offsets):

    r_pos = pos_data[:, 0, :]
    
    real_offset = tgt_offsets.copy()
    new_quats = data
    parents = [-1] * tgt_offsets.shape[0]  

    for idx, chain in enumerate(kinematic_chain):
        for index, i in enumerate(chain[1:]):
            parents[i] = chain[index]       
    
    for j in range(joints_num):
        if parents[j] != -1:
            real_offset[j, :] = real_offset[j, :] * np.linalg.norm(pos_data[0, j, :] - pos_data[0, parents[j], :])
    real_offset[0, :] = np.zeros((1, 3), np.float32)
    
    new_pos = real_offset[None, ...].repeat(data.shape[0], axis=0)
    new_pos[:, 0, :] = r_pos
    # offset, pos, quats, parents, names
    anim = Anim(new_quats, new_pos, real_offset, parents, None)

    save_bvh(path, anim, 1/20)

def output_bvh_from_real_rot(path, data, joints_num, kinematic_chain, tgt_offsets, names=None):

    device = data.device
    r_rot_quat, r_pos = recover_root_rot_pos(data) # Seq 4 , 
    cont6d_params = data[..., 4 + (joints_num-1) * 3:].reshape(data.shape[:-1] + (joints_num, 6))
    new_quats = cont6d2q(cont6d_params) # Seq J 4
    new_quats[..., 0, :] = qmultipy(r_rot_quat, new_quats[..., 0, :])
    real_offset = tgt_offsets.copy()

    parents = [-1] * tgt_offsets.shape[0]  

    for idx, chain in enumerate(kinematic_chain):
        for index, i in enumerate(chain[1:]):
            parents[i] = chain[index]       
    
    # for j in range(joints_num):
    #     if parents[j] != -1:
    #         real_offset[j, :] = real_offset[j, :] * np.linalg.norm(pos_data[0, j, :] - pos_data[0, parents[j], :])
    real_offset[0, :] = np.zeros((1, 3), np.float32)
    
    new_pos = real_offset[None, ...].repeat(data.shape[0], axis=0)
    new_pos[:, 0, :] = r_pos
    # offset, pos, quats, parents, names
    anim = Anim(new_quats, new_pos, real_offset, parents, names)

    save_bvh(path, anim, 1/20)


def get_ee_id_by_names(joint_names, ees=['RightToeBase', 'LeftToeBase', 'LeftFoot', 'RightFoot']):
    ee_id = []
    for i, name in enumerate(joint_names):
        if ':' in name:
            joint_names[i] = joint_names[i].split(':')[1]
    for i, ee in enumerate(ees):
        ee_id.append(joint_names.index(ee))
    return ee_id


def get_foot_contact(ref_motion, ee_ids, ref_height=None, thr=0.003):

    ee_pos = ref_motion[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    if ref_height is not None:
        ee_velo = torch.tensor(ee_velo) / ref_height
    else:
        ee_velo = torch.tensor(ee_velo)
    ee_velo_norm = torch.norm(ee_velo, dim=-1)
    contact = ee_velo_norm < thr
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact.numpy()

def get_foot_contact_by_vel_acc(ref_motion, ee_ids, ref_height=None, thr=0.003, use_window=False):

    ee_pos = ref_motion[:, ee_ids, :]
    # for i in range(ee_pos.shape[-2]):
    #     ee_pos[..., i, 1] = Butterworth(ee_pos[..., i, 1], 1/20, 3)
    
    butter_motion = ref_motion.copy()
    butter_motion[:, ee_ids, :] = ee_pos

    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    if ref_height is not None:
        ee_velo = ee_velo/ ref_height
    
    ee_y_vel = ee_velo[..., 1]
    
    ee_y_vel = torch.Tensor(ee_y_vel)
    
    ee_y_acc = ee_y_vel[1:] - ee_y_vel[:-1]
    # contact = abs(ee_y_vel[:-1]) < thr and (ee_y_acc) > 0

    contact1 = torch.where(torch.abs(ee_y_vel[:-1]) < thr, 1, 0)
    contact2 = torch.where(ee_y_acc > 0, 1, 0)
    contact = contact1.mul(contact2)
    
    extra_contact1 = torch.where(ee_y_vel[:-1] < 0, 1, 0)
    extra_contact2 = torch.where(ee_y_vel[1:] > 0, 1, 0)
    extra_contact = extra_contact1.mul(extra_contact2)
    
    contact = contact + extra_contact
    contact = torch.where(contact >= 1, 1, 0)

    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact, padding[:1, :]], dim=0)
    contact_new = contact.clone()
    
    if use_window:
        window = 3
        ee_pos = torch.Tensor(ee_pos)
        for i in range(ee_pos.shape[-2]):
            for frame in range(contact.shape[0]):
                if contact[frame, i] == 1:
                    start = 0 if frame-window < 0 else frame-window
                    end = contact.shape[0] if frame+window+1 > contact.shape[0] else frame+window+1
                    res_height = ee_pos[start:end, i, 1] - ee_pos[frame, i, 1]
                    added_contacts = torch.where(torch.abs(res_height) < 0.006, 1, 0)
                    contact_new[start:end, i] = added_contacts
            
    return contact_new.numpy(), ee_y_vel.numpy(), butter_motion


def get_foot_contact_by_vel3(ref_motion, ee_ids, ref_height=None, thr=0.005, use_butterworth=False):

    ee_pos = ref_motion[:, ee_ids, :]
    
    if use_butterworth:
        for i in range(ee_pos.shape[-2]):
            for j in range(ee_pos.shape[-1]):
                ee_pos[..., i, j] = Butterworth(ee_pos[..., i, j], 1/20, 3)
    
    butter_motion = ref_motion.copy()
    butter_motion[:, ee_ids, :] = ee_pos

    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    if ref_height is not None:
        ee_velo = ee_velo/ ref_height
    
    ee_y_vel = np.linalg.norm(ee_velo, ord=2, axis=-1)
    
    ee_y_vel = torch.Tensor(ee_y_vel)
    
    # ee_y_acc = ee_y_vel[1:] - ee_y_vel[:-1]
    # contact = abs(ee_y_vel[:-1]) < thr and (ee_y_acc) > 0

    contact1 = torch.where(ee_y_vel < thr, 1, 0)
    # contact2 = torch.where(ee_y_acc > 0, 1, 0)
    # contact = contact1.mul(contact2)
    
    # extra_contact1 = torch.where(ee_y_vel[:-1] < 0, 1, 0)
    # extra_contact2 = torch.where(ee_y_vel[1:] > 0, 1, 0)
    # extra_contact = extra_contact1.mul(extra_contact2)
    
    # contact = contact + extra_contact
    # contact = torch.where(contact >= 1, 1, 0)

    contact = contact1.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([contact, padding[:1, :]], dim=0)
    contact_new = contact.clone()

            
    return contact_new.numpy(), ee_y_vel.numpy(), butter_motion


def remove_fs(output_path, glb_motion, ref_motion, bonenames, ee_names, interp_length=5, force_on_floor=False,use_window=False, use_vel3=False, use_butterworth=False, vel3_thr=0.01, after_butterworth=False):
    ref_motion = ref_motion.copy()
    glb_motion = glb_motion.copy()
    if use_butterworth:
        for i in range(glb_motion.shape[-2]):
            for j in range(glb_motion.shape[-1]):
                glb_motion[..., i, j] = Butterworth(glb_motion[..., i, j], 1/20, 3)
    
    def softmax(x, **kw):
        softness = kw.pop("softness", 1.0)
        maxi, mini = np.max(x, **kw), np.min(x, **kw)
        return maxi + np.log(softness + np.exp(mini - maxi))

    def softmin(x, **kw):
        return -softmax(-x, **kw)
    # glb_motion: T J 3
    fid = get_ee_id_by_names(bonenames, ee_names)

    def alpha(t):
        return 2.0 * t * t * t - 3.0 * t * t + 1
    
    def lerp(a, l, r):
        return (1 - a) * l + a * r

    T = len(glb_motion)

    # fid_l, fid_r = np.array([fid[2], fid[1]]), np.array([fid[3], fid[0]]) 
    
    # foot_heights = np.minimum(glb_motion[:, fid_l, 1],
    #                           glb_motion[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
        
    foot_heights = glb_motion[..., 1].min(axis=1)

    # floor_height = softmin(foot_heights, softness=0.5, axis=0)
    floor_height = foot_heights.min()
    # print(floor_height)
    glb_motion[:, :, 1] -= floor_height
    
    if use_vel3:
        contacts, foot_vels, butter_motion = get_foot_contact_by_vel3(ref_motion, fid, thr=vel3_thr)
    else:
        contacts, foot_vels, butter_motion = get_foot_contact_by_vel_acc(ref_motion, fid, thr=0.003, use_window=use_window)

    for i, fidx in enumerate(fid):
        fixed = contacts[:, i]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb_motion[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb_motion[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb_motion[j, fidx] = avg.copy()

            # print(fixed[s - 1:t + 2])

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb_motion[s, fidx], glb_motion[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb_motion[s, fidx], glb_motion[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb_motion[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb_motion[s, fidx], glb_motion[l, fidx])
                glb_motion[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb_motion[s, fidx], glb_motion[r, fidx])
                glb_motion[s, fidx] = ritp.copy()
    
    if after_butterworth:
        for i in range(glb_motion.shape[-2]):
            for j in range(glb_motion.shape[-1]):
                glb_motion[..., i, j] = Butterworth(glb_motion[..., i, j], 1/20, 2.5)
    
    targetmap = {}
    for j in range(glb_motion.shape[1]):
        targetmap[j] = glb_motion[:, j]

    return glb_motion, foot_vels, contacts, butter_motion

def fit_joints_bvh(path, initial_data, joint_num, skeleton, real_offset, glb, names=None, use_lbfgs=False, iter_num=100):

    glb = torch.Tensor(glb)

    ik_solver = InverseKinematics_hmlvec(initial_data, joint_num, skeleton, real_offset, glb, use_lbfgs=use_lbfgs)

    # print('Removing Foot sliding')
    
    if iter_num is None:
        loss = 999999
        while loss > 2e-5:
            ik_solver.step()
      
    else:
        for i in tqdm(range(iter_num)):
            ik_solver.step()
  

    cont6d = ik_solver.cont6d_params
    r_pos = ik_solver.r_pos
    r_rot_quat = ik_solver.r_rot_quat
    
    r_rot_quat = qnorm(r_rot_quat)
    joint_quats = cont6d2q(cont6d)
    joint_quats[..., 0, :] = qmultipy(r_rot_quat, joint_quats[..., 0, :])
    joint_quats = joint_quats.detach().cpu().numpy()
    parents = skeleton._parents 
    
    real_offset = real_offset.copy()
    real_offset[0, :] = np.zeros((1, 3), np.float32)
    new_pos = real_offset[None, ...].repeat(joint_quats.shape[0], axis=0)
    new_pos[:, 0, :] = r_pos.detach().cpu().numpy()
    # offset, pos, quats, parents, names
    anim = Anim(joint_quats, new_pos, real_offset, parents, names)

    save_bvh(path, anim, 1/20)

def fit_joints_bvh_quats(path, real_offset, glb):

    anim = read_bvh(path)
    quats = torch.Tensor(anim.quats)
    pos = torch.Tensor(anim.pos)
    parents = anim.parents
    glb = torch.Tensor(glb)
  
    ik_solver = InverseKinematics_quats(quats, pos, parents, glb)

    # print('Removing Foot sliding')
    for i in tqdm(range(50)):
        ik_solver.step()

    cont6d = ik_solver.cont6d.detach().cpu().numpy()
    joint_quats = wrap(cont6d2q, cont6d)
    real_offset = real_offset.copy()
    real_offset[0, :] = np.zeros((1, 3), np.float32)
    new_pos = anim.pos
    anim = Anim(joint_quats, new_pos, real_offset, parents, anim.bones)

    save_bvh(path, anim, 1/20)


def Butterworth(indata, delataTimeinsec, CutOff):
    """
    :param indata:
    :param delataTimeinsec: 1/framerate
    :param CutOff: 2
    :return:
    """
    if indata is None: return None
    if CutOff == 0: return indata
    Samplingrate = 1 / delataTimeinsec
    dF2 = len(indata)-1
    Dat2 = np.zeros(dF2 + 4)
    data = indata.copy()
    for r in range(dF2):
        Dat2[2 + r] = indata[r]
    Dat2[1] = Dat2[0] = indata[0]
    Dat2[dF2 + 3] = Dat2[dF2 + 2] = indata[dF2]
    pi = 3.14159265358979
    wc = np.tan(CutOff * pi/Samplingrate)
    k1 = 1.414213562 * wc
    k2 = wc * wc
    a = k2/(1 + k1 + k2)
    b = 2 * a
    c = a
    k3 = b / k2
    d = -2 * a + k3
    e = 1 - (2 * a) - k3

    DatYt = np.zeros(dF2 + 4)
    DatYt[1] = DatYt[0] = indata[0]
    for s in range(2, dF2 +2):
        DatYt[s] = a * Dat2[s] + b * Dat2[s - 1] + \
                   c * Dat2[s - 2] + d * DatYt[s - 1] + e * DatYt[s - 2]
    DatYt[dF2 + 3]  = DatYt[dF2 + 2] = DatYt[dF2 + 1]

    DatZt = np.zeros(dF2 + 2)
    DatZt[dF2] = DatYt[dF2 + 2]
    DatZt[dF2 + 1] = DatYt[dF2 + 3]
    for t in range(-dF2 + 1, 1):
        DatZt[-t] = a * DatYt[-t + 2] + b * DatYt[-t + 3] + \
                    c * DatYt[-t + 4] + d * DatZt[-t + 1] + e * DatZt[-t + 2]

    for p in range(dF2):
        data[p] = DatZt[p]
    return data


if __name__ == '__main__':
    
    a = np.ones((178,))
    b = Butterworth(a, 1/20, 2)
    print(b.shape)

    # anim_with_end = read_bvh('./D1_010_KAN01_002.bvh', end_sites=True)

    # # anim = read_bvh('./D1_010_KAN01_002.bvh')
    # anim = read_bvh('./fallAndGetUp3_subject1_tpose.bvh')
    # # anim = read_bvh('./clip_retar_our_6_clean_smooth2_tpose_revise.bvh')
    # kinematic_chain, n_raw_offsets, real_offsets = extract_chains(anim)
    # n_raw_offsets = torch.Tensor(n_raw_offsets)
    # real_offsets = torch.Tensor(real_offsets)

    # res = wrap(quat_fk, anim.quats, anim.pos, anim.parents)

    # positions = res[1]

    # from skeleton import *
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # skel = Skeleton(n_raw_offsets, kinematic_chain, device)

    # data, global_positions, positions, l_velocity = process_file(positions, [5, 1, 18, 14], 0.002)

    # rec_positions = recover_from_ric(torch.Tensor(data), 22)    

    # output_bvh('./test.bvh', torch.Tensor(data).to(device), 22, kinematic_chain, real_offsets)
    # from plot_3d_global import plot_3d_motion, imageio
    # lafan1 = plot_3d_motion([positions[:500], None, kinematic_chain, 'title'])
    # lafan1_rec = plot_3d_motion([rec_positions[:500].cpu().numpy(), None, kinematic_chain, 'title'])
    #
    # imageio.mimsave('./lafan1.gif', np.array(lafan1), fps=30)
    # imageio.mimsave('./lafan1_rec.gif', np.array(lafan1_rec), fps=30)