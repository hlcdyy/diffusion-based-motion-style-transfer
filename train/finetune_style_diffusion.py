# This code is based on https://github.com/openai/guided-diffusion
"""
finetune a diffusion-based motion prior for style transfer 
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from utils.fixseed import fixseed
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.parser_util import finetune_inpainting_style_args
from utils import dist_util
from train.training_loop import TrainInpaintingLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import creat_ddpm_ddim_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch 
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from data_loaders.tensors import collate
from data_loaders.humanml.common.bvh_utils import remove_fs, fit_joints_bvh_quats, fit_joints_bvh
from visualize.vis_utils import joints2bvh
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.common.skeleton import Skeleton


def main():
    args = finetune_inpainting_style_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)

    if args.dataset == 'humanml':
        motion_dir = './processed_data/HumanML3D/new_joint_vecs'
        if not args.style_example:
            args.style_example = 'M008551.npy'
        path = os.path.join(motion_dir, args.style_example)
    elif args.dataset == 'bandai-2_posrot':
        motion_dir = './processed_data/bandai-2/new_joint_vecs'
        if not args.style_example:
            args.style_example = 'dataset-2_walk-turn-right_feminine_018.npy'
        path = os.path.join(motion_dir, args.style_example)
    elif args.dataset == 'stylexia_posrot':
        motion_dir = './processed_data/style_xia/new_joint_vecs'
        if not args.style_example:
            args.style_example = '350angry_jumping.npy'
        path = os.path.join(motion_dir, args.style_example)

        
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    args.save_dir = os.path.join(args.save_dir, args.style_example[:-4])

    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:    
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.bandai_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES
        real_offset = paramUtil.bandai_real_offsets
        skeleton = paramUtil.bandai_kinematic_chain
        anim = Skeleton(torch.Tensor(paramUtil.bandai_raw_offsets), skeleton, dist_util.dev())
        ee_names = ["Toes_R", 'Toes_L', 'Foot_L', 'Foot_R']

    elif args.dataset == "stylexia_posrot":
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='train')
        from model.mdm_forstyledataset import StyleDiffusion
        from data_loaders.stylexia_posrot_utils import get_inpainting_mask, BVH_JOINT_NAMES
        real_offset = paramUtil.xia_real_offsets
        skeleton = paramUtil.xia_kinematic_chain
        anim = Skeleton(torch.Tensor(paramUtil.xia_raw_offsets), skeleton, dist_util.dev())
        ee_names = ["rtoes", 'ltoes', 'lfoot', 'rfoot']



    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion
    if args.use_ddim:
        model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion, timestep_respacing="ddim20")

    else:
        model, inpainting_diffusion, inpainting_diffusion_ddpm = creat_ddpm_ddim_diffusion(args, ModelClass=StyleDiffusion)

    model.to(dist_util.dev())
    model.eval()

    max_frames = 196 if args.dataset in ['kit', 'humanml', 'bandai-1_posrot', 'bandai-2_posrot'] else 60
    max_frames = 76 if args.dataset == "stylexia_posrot" else max_frames  
    
    input_motions, m_length = data.dataset.t2m_dataset.process_np_motion(path)
    input_motions = torch.Tensor(input_motions.T).unsqueeze(1).unsqueeze(0)
    input_motions = input_motions.to(dist_util.dev()) 

    collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': m_length}] * 1
    if args.dataset == 'humanml':
        caption = 'a figure skips in a circle'
        style_label = 'happily'
        
    elif args.dataset == 'bandai-2_posrot':
        if not args.style_example:
            caption = 'a person walks turn right normal'
            style_label = 'feminine'
        else:
            contents = args.style_example.split("_")[-3].split("-")
            style_label = args.style_example.split("_")[-2]
            contents[0] += "s"
            contents = " ".join(contents)
            caption = 'a person ' +  contents + " normal"
    elif args.dataset == 'stylexia_posrot':
        if not args.style_example:
            caption = 'a person is jumping neutral'
            style_label = 'angry'
        else:
            contents = args.style_example.split("_")[-1][:-4]
            style_label = args.style_example.split("_")[0][3:]
            caption = 'a person is ' + contents + " neutral"


    texts = [caption] * 1
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)    
    model_kwargs['y']['inpainted_motion'] = input_motions 
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())
    model_kwargs['y']['scale'] = torch.ones(1, device=dist_util.dev()) * 2.5

    if args.use_ddim:
        stop_timesteps = int(900/1000 * 20)
        sample_fn = inpainting_diffusion.ddim_sample_loop
    else:
        stop_timesteps = 900
        sample_fn = inpainting_diffusion.p_sample_loop
    with torch.no_grad():
        if args.dataset == 'humanml':
            sample_fn = inpainting_diffusion_ddpm.p_sample_loop
            net = model.controlmdm
            sample = sample_fn(
            net,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            )
            
            img = torch.randn(*sample.shape, device=dist_util.dev())
            my_t = torch.ones([sample.shape[0]], device=dist_util.dev(), dtype=torch.long) * 301
            noised_img = inpainting_diffusion_ddpm.q_sample(sample, my_t, img, model_kwargs=model_kwargs)
            noised_denorm = data.dataset.t2m_dataset.inv_transform(noised_img.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
            noised_denorm = recover_from_ric(noised_denorm, 22)
            noised_denorm = noised_denorm.view(-1, *noised_denorm.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
            neutral_noised_motion = noised_denorm[0].transpose(2, 0, 1)[:m_length]
            save_bvh = 'generated_noised_neutral_motion.bvh'
            bvh_save_path = os.path.join(args.save_dir, save_bvh)
            joints2bvh(bvh_save_path, neutral_noised_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=150)
        

            sample_generated_content = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
            sample_generated_content = recover_from_ric(sample_generated_content, 22)
            sample_generated_content = sample_generated_content.view(-1, *sample_generated_content.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

            ref_motion = sample_generated_content[0].transpose(2, 0, 1)[:m_length]
            
            ref_motion, _, _, _ = remove_fs("", ref_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, interp_length=2, use_window=False, use_vel3=True, vel3_thr=0.03, after_butterworth=True)
            save_bvh = 'generated_neutral_motion.bvh'
            bvh_save_path = os.path.join(args.save_dir, save_bvh)
            joints2bvh(bvh_save_path, ref_motion, real_offset, skeleton, names=BVH_JOINT_NAMES, num_smplify_iters=150)

            new_index = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
            fit_joints_bvh_quats(bvh_save_path, real_offset, ref_motion[..., new_index, :])

        else:
            sample_fn = inpainting_diffusion_ddpm.p_sample_loop
            stop_timesteps = 900
            net = model.motion_enc.mdm_model
            sample = sample_fn(
            net,
            (1, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            stop_timesteps=stop_timesteps,
            dump_all_xstart=True,
            )
            sample = sample[-1]
            
            joint_num = 21 if args.dataset == 'bandai-2_posrot' else 20
            img = torch.randn(*sample.shape, device=dist_util.dev())
            my_t = torch.ones([sample.shape[0]], device=dist_util.dev(), dtype=torch.long) * 701
            noised_img = inpainting_diffusion_ddpm.q_sample(sample, my_t, img, model_kwargs=model_kwargs)
            noised_denorm = data.dataset.t2m_dataset.inv_transform(noised_img.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
            noised_denorm_np = recover_from_ric(noised_denorm, joint_num)
            noised_denorm_np = noised_denorm_np.view(-1, *noised_denorm_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
            neutral_noised_motion = noised_denorm_np[0].transpose(2, 0, 1)[:m_length]
            save_bvh = 'generated_noised_neutral_motion.bvh'
            bvh_save_path = os.path.join(args.save_dir, save_bvh)
            with torch.enable_grad():
                fit_joints_bvh(bvh_save_path, noised_denorm[0, 0, :m_length, :], joint_num, anim, real_offset, neutral_noised_motion, BVH_JOINT_NAMES, iter_num=100)
            save_file = 'generated_noised_neutral_motion{:02d}.mp4'.format(0)
            animation_save_path = os.path.join(args.save_dir, save_file)
            gt_frames_per_sample = {}
            plot_3d_motion(animation_save_path, skeleton, neutral_noised_motion, title="neutral noised motion",
                dataset=args.dataset, fps=20, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
           

            sample_generated_content = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float() # B 1 T J
            sample_generated_content_np = recover_from_ric(sample_generated_content, joint_num)
            sample_generated_content_np = sample_generated_content_np.view(-1, *sample_generated_content_np.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

            ref_motion = sample_generated_content_np[0].transpose(2, 0, 1)[:m_length]
            
            ref_motion, _, _, _ = remove_fs("", ref_motion, ref_motion, BVH_JOINT_NAMES, ee_names, force_on_floor=False, interp_length=3, use_window=False, use_vel3=True, vel3_thr=0.03, after_butterworth=True)
            
            save_bvh = 'generated_neutral_motion.bvh'
            bvh_save_path = os.path.join(args.save_dir, save_bvh)
            with torch.enable_grad():
                fit_joints_bvh(bvh_save_path, sample_generated_content[0, 0, :m_length, :], joint_num, anim, real_offset, ref_motion, BVH_JOINT_NAMES)

            save_file = 'generated_neutral_motion{:02d}.mp4'.format(0)
            animation_save_path = os.path.join(args.save_dir, save_file)
            gt_frames_per_sample = {}
            plot_3d_motion(animation_save_path, skeleton, ref_motion, title=" ",
                dataset=args.dataset, fps=20, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
            


    model.train()
    data_style = ((sample.detach(), model_kwargs), )

    class InpaintingDataLoader(object):
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            for motion, cond in super().__getattribute__('data').__iter__():
                if args.inpainting_mask != "":
                    cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).to(motion.device)
        
                if args.dataset == 'humanml':
                    if args.semantic_guidance:
                        for i in range(len(cond["y"]["tokens"])):
                            token_sent = cond["y"]["tokens"][i]
                            tokens = token_sent.split("_")
                            verb_idx = [i-1 for i, token in enumerate(tokens) if '/VERB' in token]
                            caption = cond['y']['text'][i].split(" ")
                            for j, idx in enumerate(verb_idx):
                                caption.insert(idx + 1 + j, style_label)
                            caption = " ".join(caption)
                            cond['y']['text'][i] = caption
                elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
                    if args.semantic_guidance:
                        for i in range(len(cond["y"]["text"])):
                            caption = cond["y"]["text"][i].split(" ")
                            caption.pop(-1)
                            caption.insert(-1, style_label)
                            caption = " ".join(caption)
                            cond['y']['text'][i] = caption
                yield motion, cond
        
        def __getattribute__(self, name):
            return super().__getattribute__('data').__getattribute__(name)
        
        def __len__(self):
            return len(super().__getattribute__('data'))
        
    data = InpaintingDataLoader(data)

    # FIXME explain
    class InpaintingTrainLoop(TrainInpaintingLoop):
        def _load_optimizer_state(self):
            pass
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_enc()) / 1000000.0))
    print("Training...")
    InpaintingTrainLoop(args, train_platform, model, data, diffusion=inpainting_diffusion, style_data=data_style).run_loop()
    train_platform.close()

    model = model.eval()
    sample_fn = inpainting_diffusion.ddim_sample_loop
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())
    
    sample = sample_fn(
            model,
            sample.detach().shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps= int(700/1000 * 20),  # 0 is the default value - i.e. don't skip any step
            init_image=sample.detach(),
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn_with_grad=False,
            pred_xstart_in_graph=False,
            dump_all_xstart=False,
        )

 
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample.shape[1] == 263 or 199 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        sample = sample.cpu().numpy() 

        
    motion = sample[0].transpose(2, 0, 1)[:m_length]
    save_file = 'style_example_rec{:02d}.mp4'.format(0)
    animation_save_path = os.path.join(args.save_dir, save_file)
    gt_frames_per_sample = {}
    plot_3d_motion(animation_save_path, skeleton, motion, title=" ",
                dataset=args.dataset, fps=20, vis_mode='gt',
                gt_frames=gt_frames_per_sample.get(0, []))
    
   

if __name__ == "__main__":
    main()
