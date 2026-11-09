import torch
import pickle
import lmdb




def log_and_print(message, file):
    print(message)
    file.write(message + "\n")




def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total parameters: {total:,}")
    #print(f"Trainable parameters: {trainable:,}")
    return total, trainable

class AuthorEmbLMDB:
    def __init__(self, path):
        self.path = path
        self._pickle = pickle
        self.env = None 

    def _require_env(self):
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=True)
        return self.env

    def __getitem__(self, author_id: str) -> torch.Tensor:
        env = self._require_env()
        with env.begin() as txn:
            data = txn.get(author_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"author_id '{author_id}' not found in LMDB")
            arr = self._pickle.loads(data)
            return torch.tensor(arr, dtype=torch.float)
        

def validate_config(cfg: dict):
    # ---------- 1. hypernet + empty target_layers ----------
    styler = cfg.get("styler_config", {})
    use_hypernet = styler.get("use_hypernet", False)

    if use_hypernet:
        hyper = styler.get("hypernet_config", {})
        target_layers = hyper.get("target_layers", [])

        if not target_layers:
            raise ValueError(
                "use_hypernet=True but hypernet_config.target_layers is empty."
            )

        for layer in target_layers:
            if layer not in ["selfattn", "crossattn", "ffn"]:
                raise ValueError("target_layers is consists of combinations of [selfattn, crossattn, ffn]")

        # ---------- 2. selfattn / crossattn requires num_prefix ----------
        if any(t in ("selfattn", "crossattn") for t in target_layers):
            num_prefix = hyper.get("num_prefix", None)
            if num_prefix is None or num_prefix == 0:
                raise ValueError(
                    "target_layers contains 'selfattn' or 'crossattn' "
                    "but hypernet_config.num_prefix is null or empty."
                )
        # ---------- 3. ffn requires rank ----------
        if any(t in ("ffn") for t in target_layers):
            num_prefix = hyper.get("rank", None)
            if num_prefix is None or num_prefix == 0:
                raise ValueError(
                    "target_layers contains 'ffn' "
                    "but hypernet_config.rank is null or empty."
                )
    # ---------- 4. train_config.freeze validation ----------
    train_cfg = cfg.get("train_config", {})
    freeze_list = train_cfg.get("freeze", [])

    allowed_freeze = {"encoder", "all"}
    invalid = set(freeze_list) - allowed_freeze
    if invalid:
        raise ValueError(
            f"train_config.freeze contains invalid values: {invalid}. "
            f"Allowed values are {allowed_freeze}."
        )
