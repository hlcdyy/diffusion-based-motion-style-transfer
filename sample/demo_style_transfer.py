# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import json
from utils.fixseed import fixseed
from utils.parser_util import eval_inpainting_style_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import load_model_wo_controlmdm, creat_serval_diffusion,load_model_wo_moenc
import torch 
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
import data_loaders.humanml.utils.paramUtil as paramUtil
import shutil
import numpy as np

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.bvh_utils import output_bvh_from_real_rot, remove_fs, fit_joints_bvh
from visualize.vis_utils import joints2rotation, joints2bvh
from scipy import io


def main():
    output_bvh = False
    args = eval_inpainting_style_args()
    fixseed(args.seed)
    out_dir = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['humanml',"bandai-1_posrot", "bandai-2_posrot"] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    
    if out_dir == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'style_transfer_from_stylexample_{}_to_contentmotion_{}_seed{}'.format(name, args.input_content[:-4], args.seed))
        if args.input_text != '':
            out_path += '_' + args.input_text.replace(' ', '_').replace('.', '')
    else:
        out_path = os.path.join(args.output_dir,
                                'style_transfer_from_stylexample_{}_to_contentmotion_{}_seed{}'.format(name, args.input_content[:-4], args.seed))
        if args.input_text != '':
            out_path += '_' + args.input_text.replace(' ', '_').replace('.', '')
    


    print("creating data loader...")
    args.batch_size = args.num_samples
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
        from model.mdm_forstyledataset import StyleDiffusion
        if args.dataset == 'stylexia_posrot':
            from data_loaders.stylexia_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES
        else:
            from data_loaders.bandai_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES

    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='test')
    
    

    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion

    use_ddim = True

    if use_ddim:
        model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")
    else:
        model, inpainting_diffusion, normal_diffusion = creat_serval_diffusion(args, ModelClass=StyleDiffusion)
    inpainting_style_dict = torch.load(args.model_path, map_location='cpu')
    print("load style diffusion model: {}".format(args.model_path))
    if args.dataset == 'humanml':
        load_model_wo_controlmdm(model, inpainting_style_dict)
    else:
        load_model_wo_moenc(model, inpainting_style_dict)

    model.to(dist_util.dev())
    model.eval()

    if args.dataset == 'stylexia_posrot':
        motion_dir = './processed_data/style_xia/new_joint_vecs'
        if not args.style_example:
            args.style_example = '350angry_jumping.npy'
        path = os.path.join(motion_dir,args.style_example)
        joint_num = 20
 
    else:
        motion_dir = './processed_data/bandai-2/new_joint_vecs'
        if not args.style_example:
            args.style_example = 'dataset-2_walk-turn-right_feminine_018.npy'
        path = os.path.join(motion_dir, args.style_example)
        joint_num = 21

    if args.dataset in ['humanml']:
        skeleton = paramUtil.t2m_kinematic_chain
        real_offset = paramUtil.smpl_real_offsets
        ee_names = ["R_Ankle", "L_Ankle", "L_Foot", "R_Foot"]
        anim = Skeleton(torch.Tensor(paramUtil.smpl_raw_offsets), skeleton, dist_util.dev())
    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        skeleton = paramUtil.bandai_kinematic_chain
        real_offset = paramUtil.bandai_real_offsets
        ee_names = ["Toes_R", 'Toes_L', 'Foot_L', 'Foot_R']
        anim = Skeleton(torch.Tensor(paramUtil.bandai_raw_offsets), skeleton, dist_util.dev())
    elif args.dataset == 'stylexia_posrot':
        skeleton = paramUtil.xia_kinematic_chain
        real_offset = paramUtil.xia_real_offsets
        ee_names = ["rtoes", 'ltoes', 'lfoot', 'rfoot']
        anim = Skeleton(torch.Tensor(paramUtil.xia_raw_offsets), skeleton, dist_util.dev())
    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    input_motions, style_m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 

    sample_t2m_denorm = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
    sample_t2m_np = recover_from_ric(sample_t2m_denorm, joint_num)
    sample_t2m_np = sample_t2m_np.view(-1, *sample_t2m_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    #  B J 3 T
    if args.input_text != '':
        texts = [args.input_text] * args.num_samples
    else:
        contents = args.input_content.split("_")[-1][:-4]
        style_example_name = args.model_path.split("/")[-2]
        style_label =  style_example_name.split("_")[0][3:]
        caption = 'A person is ' + contents + " " + style_label
        texts = [caption] * args.num_samples

    
    collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': style_m_length}] * 1 # m_length
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    
    if args.dataset == 'humanml':
        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': style_m_length}] * 1
        caption = 'a figure skips in a circle'

        if args.input_text != '':
            texts = [args.input_text] * args.num_samples
        else:
            contents = args.input_content.split("_")[-1][:-4]
            style_example_name = args.model_path.split("/")[-2]
            style_label =  style_example_name.split("_")[0][3:]
            caption = 'A person is ' + contents + " " + style_label 
            texts = [caption] * args.num_samples

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)    
        model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5
        
        sample_fn = normal_diffusion.p_sample_loop

        sample_t2m = sample_fn(
            model.controlmdm.motion_enc.mdm_model,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            # init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        m_length = style_m_length
    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
        if args.dataset == "stylexia_posrot":
            path = os.path.join(motion_dir, args.input_content)

            
            


        sample_t2m, m_length = data.dataset.t2m_dataset.process_np_motion(path)
        sample_t2m = torch.Tensor(sample_t2m.T).unsqueeze(1).unsqueeze(0)
        sample_t2m = sample_t2m.to(dist_util.dev())

        
        if args.input_text != '':
            texts = [args.input_text] * args.num_samples
        else:
            contents = args.input_content.split("_")[-1][:-4]
            style_example_name = args.model_path.split("/")[-2]
            style_label =  style_example_name.split("_")[0][3:]
            caption = 'A person is ' + contents + " " + style_label 
            texts = [caption] * args.num_samples

            
        collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1 # m_length
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)
     
    if model.data_rep == 'hml_vec':
        sample_t2m_denorm = data.dataset.t2m_dataset.inv_transform(sample_t2m.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
        sample_t2m_np = recover_from_ric(sample_t2m_denorm, joint_num)
        sample_t2m_np = sample_t2m_np.view(-1, *sample_t2m_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        #  B J 3 T
        save_bvh = 'input_content_motion.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        # output_bvh_from_real_rot(bvh_save_path, sample_t2m_np[0][0][:m_length], joint_num, skeleton, real_offset, BVH_JOINT_NAMES)
        
        ref_motion = sample_t2m_np[0].transpose(2, 0, 1)[:m_length]
        if args.dataset in ['humanml']:
            ref_motion, _, _, _ = remove_fs("", ref_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False,use_vel3=True, vel3_thr=0.02, after_butterworth=True)
            
        if args.dataset != 'humanml':
            fit_joints_bvh(bvh_save_path, sample_t2m_denorm[0, 0, :m_length, :], joint_num, anim, real_offset, ref_motion, BVH_JOINT_NAMES)
     


    mask = model_kwargs['y']['mask']

    all_lengths = []
    all_motions = []
    all_text = []
    all_hml_motions = []    

    model_kwargs['y']['inpainted_motion'] = sample_t2m 
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, sample_t2m.shape)).float().to(dist_util.dev())
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')
        
        if use_ddim:
            sample_fn = inpainting_diffusion.ddim_sample_loop
            skip_timesteps = int(args.skip_steps/args.diffusion_steps * 20)
        else:
            sample_fn = inpainting_diffusion.p_sample_loop
            skip_timesteps = args.skip_steps
        
        if args.dataset == 'humanml':
            dump_all_xstart = False
        else:
            dump_all_xstart = True
        sample = sample_fn(
            model,
            sample_t2m.shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps, 
            init_image=sample_t2m,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn_with_grad=False,
            pred_xstart_in_graph=False,
            dump_all_xstart=dump_all_xstart,
        )
        if dump_all_xstart:
            sample = sample[-5]
        else:
            sample = sample

        if model.data_rep == 'hml_vec':
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            all_hml_motions.append(sample)
            sample = recover_from_ric(sample, joint_num)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    all_hml_motions = torch.cat(all_hml_motions, dim=0)

    
    gt_frames_per_sample = {}
    
    npy_path = os.path.join(out_path, 'results.npy') 

    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})



    print(f"saving visualizations to [{out_path}]...")

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        
        input_motions_denorm = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions_denorm, joint_num)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        
        save_bvh = 'input_style_example.bvh'
        bvh_save_path = os.path.join(out_path, save_bvh)
        sty_motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]

        if args.dataset != 'humanml':
            fit_joints_bvh(bvh_save_path, input_motions_denorm[0, 0, :style_m_length], joint_num, anim, real_offset, sty_motion, BVH_JOINT_NAMES)
    

    length = all_lengths[0]
    fs_motion = all_motions[0].transpose(2, 0, 1)[:length].copy()

    fs_motion, _, _, _  = remove_fs("", fs_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=True, after_butterworth=True, use_vel3=True, vel3_thr=0.05)
    fs_motion, _, _, _ = remove_fs("", fs_motion, fs_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=True, after_butterworth=True, use_vel3=True, vel3_thr=0.05)

    save_bvh = 'out_transferred_motion.bvh'
    bvh_save_path = os.path.join(out_path, save_bvh)
    if args.dataset != 'humanml':
        fit_joints_bvh(bvh_save_path, all_hml_motions[0, 0, :length], joint_num, anim, real_offset, fs_motion, BVH_JOINT_NAMES)
    



    sample_i = 0
    rep_files = []
    caption = 'Input Content Motion'
    length = model_kwargs['y']['lengths'][0]
    motion = sample_t2m_np[0].transpose(2, 0, 1)[:length]
    save_file = 'input_content_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
    
    caption = 'Input Style Motion'
    motion = input_motions[0].transpose(2, 0, 1)[:style_m_length]
    save_file = 'input_style_motion{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(out_path, save_file)
    rep_files.append(animation_save_path)
    print(f'[({0}) "{caption}" | -> {save_file}]')
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                dataset=args.dataset, fps=fps, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))


    for rep_i in range(args.num_repetitions):
        
        caption = all_text[rep_i*args.batch_size + sample_i]
        if args.guidance_param == 0:
            caption = "style transferred motion"
        else:
            caption = 'style transferred motion: {}'.format(caption)
        length = all_lengths[rep_i*args.batch_size + sample_i]
        # motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
        motion = fs_motion
        save_file = 'output_transferred_motion{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
        animation_save_path = os.path.join(out_path, save_file)
        rep_files.append(animation_save_path)
        print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                        gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','))
        
        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + 1} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
        
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')
   

if __name__ == "__main__":
    main()
