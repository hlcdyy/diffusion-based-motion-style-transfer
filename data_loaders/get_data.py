from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_style_collate

def get_dataset_class(name):
   
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
   

    elif name in ['bandai-1_posrot', 'bandai-2_posrot', 'stylexia_posrot']:
        from data_loaders.humanml.data.dataset import StyleDataset
        return StyleDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml"]:
        return t2m_collate
    if name in ['bandai-1', 'bandai-2', 'bandai-1_posrot', 'bandai-2_posrot', 'stylexia_posrot']:
        return t2m_style_collate

    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml"]:
        # The num_frames will be ignored when used humanML and Kit dataset
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name in ["bandai-1", "bandai-2", "bandai-1_posrot", "bandai-2_posrot", "stylexia_posrot"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, dataset_name=name)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', shuffle=True):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

