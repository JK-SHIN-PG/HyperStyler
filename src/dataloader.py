import torch

def collator_cycle_reconstruction(
    *,
    batch,
    tokenizer,
    max_length_src,
    max_length_tgt,
    ignore_idx=-100,
):
    '''
        'author_id':        Value('string'),
        'texts':            Value('string'),
        'paraphrased_text': Value('string'),
        'style_embs':       Sequence(Value('float32')),
    '''

    input_texts = []
    for x in batch:
        #print("paraphrase :",x['paraphrase'][0])
        #print("text :",x['text'])
        if isinstance(x['paraphrase'], list):
            para = x['paraphrase'][0]
        else:
            para = x['paraphrase']
        input_texts.append(para.lower())
        input_texts.append(x['text'])

    output_texts = []
    for x in batch:
        if isinstance(x['paraphrase'], list):
            para = x['paraphrase'][0]
        else:
            para = x['paraphrase']
        output_texts.append(x['text'])
        output_texts.append(para.lower())
        
    
    inputs = tokenizer(
        input_texts,
        max_length=max_length_src,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )

    labels = tokenizer(
        output_texts,
        max_length=max_length_tgt,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )['input_ids']
    labels[labels == tokenizer.pad_token_id] = ignore_idx

    retval = inputs
    retval['labels'] = labels

    return retval


def collator_for_training(
    *,
    batch,
    tokenizer,
    max_length_src,
    max_length_tgt,
    ignore_idx=-100,
    author_embs_dict,
    use_stylonavigator,
):
    """
    batch element schema:
        'author_id':        str
        'text':             str
        'paraphrase': str
        'style_embs': # not use
        'idx' 
    """
    input_texts = [
    x['paraphrase'][0] if isinstance(x['paraphrase'], list) else x['paraphrase']
    for x in batch
    ]

    output_texts = [x["text"] for x in batch]

    inputs = tokenizer(
        input_texts,
        max_length=max_length_src,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    labels = tokenizer(
        output_texts,
        max_length=max_length_tgt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    

    labels[labels == tokenizer.pad_token_id] = ignore_idx
    
    retval = dict(inputs)
    retval["labels"] = labels

    # --- style embedding padding ---
    style_embeddings = []
    max_num = 0
    for x in batch:
        embs = author_embs_dict[x["author_id"]]
        style_embeddings.append(embs)
        max_num = max(max_num, embs.shape[0])

    padded_embs = []
    masks = []

    for emb in style_embeddings:
        num_pads = max_num - emb.shape[0]
        if num_pads > 0:
            padding = torch.zeros(num_pads, emb.shape[1], dtype=torch.float)
            emb_padded = torch.cat([emb, padding], dim=0)
        else:
            emb_padded = emb

        mask = torch.cat(
            [torch.ones(emb.shape[0], dtype=torch.float),
             torch.zeros(num_pads, dtype=torch.float)]
        )

        padded_embs.append(emb_padded)
        masks.append(mask)

    if use_stylonavigator:
        idx_labels = [x['idx'] for x in batch]
        retval["idx_labels"] = torch.tensor(idx_labels).long()
    
    retval["style_embs"] = torch.stack(padded_embs)  # [B, max_num, 768]
    retval["style_masks"] = torch.stack(masks)             # [B, max_num]
   
    return retval



def collator_for_self_distillation(
    *,
    batch,
    tokenizer,
    max_length_src,
    max_length_tgt,
    ignore_idx=-100,
    author_embs_dict,
    use_stylonavigator,
):
    """
    batch element schema:
        'author_id':        str
        'text':             str
        'paraphrase': str
        'style_embs': # not use
        'idx' 
    """

    input_texts = [x['text']  for x in batch]

    idx_labels = []
    output_texts = []
    for x in batch:
        output_texts.append(x['output'])
        # score = torch.tensor(x['pass'])* torch.tensor(x['joint'])
        # idx = torch.argmax(score)
        # output_texts.append(x['output'][idx])
        # pass_t = torch.tensor(x['pass'], dtype=torch.bool)          # (N,)
        # cossim_t = torch.tensor(x['cossim_toward'])          # (N, K)
        # cossim_score, _ = cossim_t.max(dim=1) 
        # cossim_score = cossim_score.masked_fill(~pass_t, float('-inf'))
        # print(cossim_score)
        # idx = torch.argmax(cossim_score)
        # output_texts.append(x['output'][idx])
        # arg_idx = torch.argmax(cossim_t[idx])
        # idx_labels.append(arg_idx)
        #print(cossim_score[idx])
        # score = torch.tensor(x['pass']) * torch.tensor(x['sim']) * torch.tensor(x['away'])
        # idx = torch.argmax(score)
        # output_texts.append(x['output'][idx])
        # arg_idx = torch.argmax(torch.tensor(x['cossim_toward'][idx]))
        # idx_labels.append(arg_idx)
        #print(arg_idx, torch.tensor(x['cossim_toward'][idx])) 

    inputs = tokenizer(
        input_texts,
        max_length=max_length_src,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # print(input_texts)
    # print(output_texts)
    labels = tokenizer(
        output_texts,
        max_length=max_length_tgt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]


    labels[labels == tokenizer.pad_token_id] = ignore_idx
    
    retval = dict(inputs)
    retval["labels"] = labels

    # lose_labels = copy.deepcopy(inputs['input_ids'])
    # lose_labels[lose_labels == tokenizer.pad_token_id] = ignore_idx
    # retval["lose_labels"] = lose_labels

    # --- style embedding padding ---
    style_embeddings = []
    max_num = 0
    for x in batch:
        embs = author_embs_dict[x["target_author_id"]]
        style_embeddings.append(embs)
        max_num = max(max_num, embs.shape[0])

    padded_embs = []
    masks = []

    for emb in style_embeddings:
        num_pads = max_num - emb.shape[0]
        if num_pads > 0:
            padding = torch.zeros(num_pads, emb.shape[1], dtype=torch.float)
            emb_padded = torch.cat([emb, padding], dim=0)
        else:
            emb_padded = emb

        mask = torch.cat(
            [torch.ones(emb.shape[0], dtype=torch.float),
             torch.zeros(num_pads, dtype=torch.float)]
        )

        padded_embs.append(emb_padded)
        masks.append(mask)

    # if use_stylonavigator:
    #     retval["idx_labels"] = torch.stack(idx_labels).long()

    retval["style_embs"] = torch.stack(padded_embs)  # [B, max_num, 768]
    retval["style_masks"] = torch.stack(masks)             # [B, max_num]
    
    #####
    # target_style_embs = torch.tensor(
    #     [x['selected_embs'] for x in batch],
    #     dtype=torch.float,
    # )
    # retval["target_style_embs"] = target_style_embs
    #####
    return retval
