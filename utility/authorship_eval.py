"""Score evaluation generations: joint, away, towards and sim per author pair.

Away and towards use LUAR embeddings (``utility/LUAR.pth``).
"""

import json

import numpy as np
from tqdm import tqdm

from utility.evaluation_metric import joint_score


def score_generations(input_path, scores_path, avg_path, device='cuda'):
    author_pair_data = {}
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            pair = f"{data['source_author']}->{data['target_author']}"
            if pair not in author_pair_data:
                author_pair_data[pair] = {
                    "source_texts": data['source_author_texts'],
                    "target_texts": data['target_author_texts'],
                    "style_transferred_texts": [],
                }
            if isinstance(data["output"], str):
                data["output"] = [data["output"]]
            author_pair_data[pair]["style_transferred_texts"].append(data["output"])

    print('Computing metrics')
    all_scores = []
    for pair_data in tqdm(list(author_pair_data.values())):
        source_texts = pair_data["source_texts"]
        target_texts = pair_data["target_texts"]
        chosen_transferred = [x[0] for x in pair_data["style_transferred_texts"]]

        if len(target_texts) > len(source_texts):
            print(
                f'Warning: target texts ({len(target_texts)}) are more than source texts ({len(source_texts)})...resizing'
            )
            target_texts = target_texts[: len(source_texts)]

        assert len(source_texts) == len(chosen_transferred) == len(target_texts)

        scores = joint_score(
            source_texts=source_texts,
            target_texts=target_texts,
            style_transferred_texts=chosen_transferred,
            device=device,
            embedding_type='uar',
        )
        all_scores.append(scores)

    with open(scores_path, "w") as f:
        json.dump(all_scores, f)

    with open(avg_path, "w") as f:
        json.dump(
            {
                "joint_score": np.mean([x[0] for x in all_scores]),
                "away_score": np.mean([x[1]["away"] for x in all_scores]),
                "towards_score": np.mean([x[1]["towards"] for x in all_scores]),
                "sim_score": np.mean([x[1]["sim"] for x in all_scores]),
                "n": len(all_scores),
            },
            f,
        )
