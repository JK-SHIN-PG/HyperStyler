"""Load and save HyperStyler state dicts (.pt or .safetensors)."""

from pathlib import Path

import torch

# T5 ties these to model.shared.weight; safetensors stores the shared tensor once.
TIED_EMBEDDINGS = ('model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight')


def load_state_dict(path, map_location='cpu'):
    path = str(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(path, device=str(map_location))
        for prefix in ('', 'hyperstyler.'):
            if prefix + 'model.shared.weight' in state_dict:
                for key in TIED_EMBEDDINGS:
                    state_dict.setdefault(prefix + key, state_dict[prefix + 'model.shared.weight'])
    else:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    # 'module.': saved from a DDP-wrapped model; 'hyperstyler.': the Hugging Face release.
    prefixes = ('module.', 'hyperstyler.')
    out = {}
    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def save_safetensors(state_dict, path):
    from safetensors.torch import save_file
    state_dict = {k: v.contiguous() for k, v in state_dict.items() if k not in TIED_EMBEDDINGS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path), metadata={'format': 'pt'})
