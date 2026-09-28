import os
import torch
import timm

RANDOM_INIT_DIR = "results/random_init"


def get_fixed_random_model(model_name, random_init_dir=RANDOM_INIT_DIR):
    """Loads a random-init model from a fixed (seeded) checkpoint, creating it on first use.

    Ensures every script and run compares against the same random baseline for a given
    model, instead of a fresh unseeded init each time.
    """
    os.makedirs(random_init_dir, exist_ok=True)
    ckpt_path = os.path.join(random_init_dir, f"{model_name}.pt")

    if not os.path.exists(ckpt_path):
        torch.manual_seed(0)
        model = timm.create_model(model_name, pretrained=False).eval()
        torch.save(model.state_dict(), ckpt_path)
    else:
        model = timm.create_model(model_name, pretrained=False).eval()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    return model
