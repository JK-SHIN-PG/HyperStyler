
from sklearn.metrics.pairwise import cosine_similarity
from mutual_implication_score import MIS
import numpy as np
import math
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
from utility.uar import get_uar_embedding
mis = None
mis_alternative = None
mis_alternative_tokenizer = None
from statistics import mean



def clean_fp_error_batch(v):
    '''Batch version of clean_fp_error using NumPy.'''
    v = np.clip(v, 0, 1)  # Clip values to [0, 1]
    v = np.where((v > 0.99999) & (v < 1.0), 1.0, v)
    v = np.where((v < 0.00001) & (v > 0.0), 0.0, v)
    return v

def clean_fp_error(v):
    #Helps ignore floating point error ie. 1.00001 or -0.0000001
    if type(v) != float:
        v = v.item()
    v = min(max(v, 0), 1)
    if v > 0.99999 and v < 1.0:
        v = 1.0
    elif v < 0.00001 and v > 0.0:
        v = 0.0
    return v

def _sim(a, b):
    return clean_fp_error(cosine_similarity(a, b).squeeze(0))

def asim(a, b):
    return 1 - np.arccos(_sim(a, b)) / np.pi

def towards_score(emb, target_emb):
    return clean_fp_error((asim(emb, target_emb) + 1.0) / 2.0)

def away_score(a, b):
    return clean_fp_error(1.0 - ((asim(a, b) + 1.0) / 2.0))


def _sim_batch(a, b):
    return clean_fp_error_batch(cosine_similarity(a, b).diagonal())

def asim_batch(a, b):
    return 1 - np.arccos(_sim_batch(a, b)) / np.pi

def towards_score_batch(emb, target_emb):
    return clean_fp_error_batch((asim_batch(emb, target_emb) + 1.0) / 2.0)

def away_score_batch(a, b):
    return clean_fp_error_batch(1.0 - ((asim_batch(a, b) + 1.0) / 2.0))

def mis_compute(a_texts, b_texts):
    global mis

    if mis is None:
        mis = MIS(device="cuda")

    scores = mis.compute(a_texts, b_texts, verbose=False, batch_size=16)
    for idx, (score, a_text, b_text) in enumerate(zip(scores, a_texts, b_texts)):
        if a_text == b_text:
            scores[idx] = 1.0

    return scores


def mis_alternative_compute(a_texts, b_texts, batch_size=32):
    global mis_alternative
    global mis_alternative_tokenizer

    if mis_alternative is None:
        mis_alternative_tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large"
        )
        mis_alternative = AutoModel.from_pretrained(
            "princeton-nlp/sup-simcse-roberta-large"
        )
        mis_alternative.eval()
        mis_alternative.to('cuda')

    texts = a_texts + b_texts

    with torch.no_grad():
        all_embeddings = []

        for i in tqdm(list(range(0, len(texts), batch_size))):
            text_batch = texts[i : i + batch_size]

            inputs = mis_alternative_tokenizer(
                text_batch, padding=True, truncation=True, return_tensors="pt"
            ).to('cuda')

            embeddings = mis_alternative(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output

            all_embeddings.append(embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        a_text_embeddings = embeddings[: len(a_texts)]
        b_text_embeddings = embeddings[len(a_texts) :]

        cos = torch.nn.CosineSimilarity(dim=1)
        sims = cos(a_text_embeddings, b_text_embeddings)
        scores = (sims + 1) / 2

    return scores.detach().cpu().numpy().tolist()


def sim_score_batch(texts, source_texts, alternative=False):
    if alternative:
        return clean_fp_error_batch(
            [
                s
                for s in mis_alternative_compute(source_texts, texts)
                if not math.isnan(s)
            ]
        )

    else:
        return clean_fp_error_batch(
            [s for s in mis_compute(source_texts, texts) if not math.isnan(s)]
        )


def sim_score(texts, source_texts, alternative=False):
    if alternative:
        return clean_fp_error(
            mean(
                [
                    s
                    for s in mis_alternative_compute(source_texts, texts)
                    if not math.isnan(s)
                ]
            )
        )

    else:
        return clean_fp_error(
            mean([s for s in mis_compute(source_texts, texts) if not math.isnan(s)])
        )

def joint_score(
    *,
    source_texts,
    target_texts,
    style_transferred_texts,
    device,
    embedding_type='uar',
    alternative_sim=False
):
    # Calculate the embedding of the texts

    if embedding_type == 'uar':
        source_emb = get_uar_embedding(source_texts, device)
        target_emb = get_uar_embedding(target_texts, device)
        st_emb = get_uar_embedding(style_transferred_texts, device)
    else:
        raise ValueError('Invalid embedding type')
    
    # Calculate the away score
    score_away = away_score(st_emb, source_emb)
    score_away_max = away_score(
        target_emb, source_emb
    )  # The maximum away we can get ever
    score_away_scaled = clean_fp_error(
        min(score_away, score_away_max) / score_away_max
    )  # Ensures we can achieve a 1.0 when reaching the away maximum

    # Calculate the toward score
    score_towards = towards_score(st_emb, target_emb)
    score_towards_min = towards_score(
        source_emb, target_emb
    )  # The minimum towards we should get automatically
    score_towards_scaled = clean_fp_error(
        max(score_towards - score_towards_min, 0) / (1.0 - score_towards_min)
    )  # Ensures we can achieve a 0.0 when scoring at or below the towards minimum

    # Calculate the sim score
    score_sim = sim_score(
        style_transferred_texts, source_texts, alternative=alternative_sim
    )
    score_sim_min = sim_score(
        target_texts, source_texts, alternative=alternative_sim
    )  # The minimum sim we should get automatically
    score_sim_scaled = clean_fp_error(
        max(score_sim - score_sim_min, 0) / (1.0 - score_sim_min)
    )  # Ensures we can achieve a 0.0 when scoring at or below the sim minimum

    # Calculate the joint score
    
    score_joint = clean_fp_error(
        (((score_away_scaled * score_towards_scaled) ** (1 / 2)) * score_sim_scaled)
        ** (1 / 2)
    )
    return clean_fp_error(score_joint), {
        'away': score_away_scaled,
        'towards': score_towards_scaled,
        'sim': score_sim_scaled,
    }
