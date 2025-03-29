import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask.float() # B max_len,


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'file_name' in notnone_batches[0]:
        file_name = [b['file_name'] for b in notnone_batches]
        cond['y'].update({'file_name':file_name})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    # collate style names
    if 'style' in notnone_batches[0]:
        style = [b['style'] for b in notnone_batches]
        cond['y'].update({'style': style})

    # collate style motion and style motion lengths (style motion as another condition)
    if 'sty_x' in notnone_batches[0]:
        sty_x = [b["sty_x"] for b in notnone_batches]
        sty_motion = collate_tensors(sty_x)
        cond.update({'sty_x': sty_motion})
        
        stylenbatch = [b['sty_lengths'] for b in notnone_batches]
        stylenbatchTensor = torch.as_tensor(stylenbatch)
        stymaskbatchTensor = lengths_to_mask(stylenbatchTensor, sty_motion.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        cond.update({"sty_y": {'mask': stymaskbatchTensor, 'lengths': stylenbatchTensor}})
        
    return motion, cond
 

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption'] 
        'tokens': b[6],  # row_token not word embeddings
        'lengths': b[5],  # motion length
        'file_name':b[7],
    } for b in batch]
    return collate(adapted_batch)


def t2m_style_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1),
        'text': b[0],
        'lengths': b[2],
        'style': b[3],
    } for b in batch]
    return collate(adapted_batch)
