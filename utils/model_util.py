from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from model.mdm_forstyledataset import DiffuseTrasnfer


    
def load_model_wo_moenc(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    assert len(unexpected_keys) == 0

    assert all([k.startswith('motion_enc.') or 
                k.startswith('input_zero.') or 
                k.startswith('output_zero.') for k in missing_keys])
    
def load_model_wo_controlmdm(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    assert len(unexpected_keys) == 0

    assert all([k.startswith('controlmdm.') for k in missing_keys])


def creat_serval_diffusion(args, ModelClass=DiffuseTrasnfer, timestep_respacing = ''):
    model = ModelClass(**get_transfer_args(args))
    diffusion1 = create_gaussian_diffusion(args, InpaintingGaussianDiffusion, timestep_respacing=timestep_respacing)
    diffusion2 = create_gaussian_diffusion(args)
    return model, diffusion1, diffusion2


def creat_ddpm_ddim_diffusion(args, ModelClass=DiffuseTrasnfer, timestep_respacing = ''):
    model = ModelClass(**get_transfer_args(args))
    diffusion1 = create_gaussian_diffusion(args, InpaintingGaussianDiffusion, timestep_respacing=timestep_respacing)
    diffusion2 = create_gaussian_diffusion(args, InpaintingGaussianDiffusion)
    return model, diffusion1, diffusion2


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        data_rep = 'hml_vec'
        njoints = 190
        nfeats = 1

    elif args.dataset == 'stylexia_posrot':
        data_rep = 'hml_vec'
        njoints = 181
        nfeats = 1


    
    if hasattr(args, 'mdm_path'):
        mdm_path = args.mdm_path
    else:
        mdm_path = ""

    if hasattr(args, 'semantic_discriminator_path'):
        semantic_discriminator_path = args.semantic_discriminator_path
    else:
        semantic_discriminator_path = ""

    if hasattr(args, 'zero_conv') and args.zero_conv:
        zero_conv = True
    else:
        zero_conv = None

    if hasattr(args, 'inpainting_model_path'):
        inpainting_model_path = args.inpainting_model_path
    else:
        inpainting_model_path = ""
        

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'mdm_path': mdm_path, 'semantic_discriminator_path': semantic_discriminator_path, 'zero_conv': zero_conv, 
            "inpainting_model_path":inpainting_model_path}


def get_transfer_args(args):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1

    
    elif args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        data_rep = 'hml_vec'
        njoints = 190
        nfeats = 1

    elif args.dataset == 'stylexia_posrot':
        data_rep = 'hml_vec'
        njoints = 181
        nfeats = 1

    
    if hasattr(args, 'mdm_path'):
        mdm_path = args.mdm_path
    else:
        mdm_path = ""

    if hasattr(args, 'semantic_discriminator_path'):
        semantic_discriminator_path = args.semantic_discriminator_path
    else:
        semantic_discriminator_path = ""
    
    if hasattr(args, 'inpainting_model_path'):
        inpainting_model_path = args.inpainting_model_path
    else:
        inpainting_model_path = ""
    

    if hasattr(args, 'zero_conv') and args.zero_conv:
        zero_conv = True
    else:
        zero_conv = None


    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'mdm_path': mdm_path, 'semantic_discriminator_path': semantic_discriminator_path, 'zero_conv': zero_conv, 
            "inpainting_model_path":inpainting_model_path}


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion, timestep_respacing = ''):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    # timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    
    print(f"number of diffusion-steps: {steps}")
    
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_sty_cons = args.lambda_sty_cons if hasattr(args, "lambda_sty_cons") else 0,
        lambda_sty_trans = args.lambda_sty_trans if hasattr(args, "lambda_sty_trans") else 0,
        lambda_cont_pers = args.lambda_cont_pers if hasattr(args, "lambda_cont_pers") else 0,
        lambda_cont_vel = args.lambda_cont_vel if hasattr(args, "lambda_cont_vel") else 0,
        lambda_diff_sty = args.lambda_diff_sty if hasattr(args, "lambda_diff_sty") else 0,
        
    )