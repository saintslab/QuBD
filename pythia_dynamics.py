import os
import re
import json
import time
import requests
from transformers import AutoModelForCausalLM

from qbdm.qbdm import measure_complexity, measure_bitplane_compression

# Environmental configuration for offline local cache priority
CACHE_DIR = os.path.expanduser("~/.cache/timm_models")
RESULTS_DIR = os.path.expanduser("./results/")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR

### Global Experiment Parameters
MODEL_NAMES = ["EleutherAI/pythia-1b"]

#["EleutherAI/pythia-70m", "EleutherAI/pythia-1b"]
BIT_DEPTHS = [8]
TOKENS_PER_STEP = 2_097_152  # every Pythia model used a fixed 2M-token batch

# Pythia checkpoints: step0, 10 log-spaced early steps, then evenly-spaced to step143000.
# We sample a log-spaced subset spanning that full range rather than all 154 checkpoints.
TARGET_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 143000]

WANDB_ENTITY = "eleutherai"
WANDB_PROJECT = "pythia"
WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
WANDB_LOSS_KEY = "train/lm_loss"
WANDB_VAL_LOSS_KEY = "validation/lm_loss"


def _wandb_graphql(query, variables):
    """Public wandb projects can be queried anonymously over GraphQL - no API key needed."""
    resp = requests.post(WANDB_GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


def find_full_training_run(model_name, entity=WANDB_ENTITY, project=WANDB_PROJECT):
    """EleutherAI's public wandb project logs every pre-emption/restart of a training job as its
    own run sharing a common group name (e.g. 'v2 70M_<id>' / 'v2-70M-deduped_<id>'). Almost all of
    these are near-empty crash artifacts from the shared training cluster; exactly one run per model
    size carries the full, continuous step history. We find it by scanning every run whose group
    matches this model's size/dedup tag and keeping the one with the most logged rows."""
    size_tag = model_name.split("/")[-1].replace("pythia-", "").replace("-deduped", "").upper()
    deduped = "deduped" in model_name
    group_prefix = f"v2-{size_tag}-deduped_" if deduped else f"v2 {size_tag}_"

    query = """
    query Q($entity:String!,$project:String!,$cursor:String,$regex:String!){
      project(name:$project, entityName:$entity){
        runs(first: 100, after: $cursor, filters: $regex){
          pageInfo { hasNextPage endCursor }
          edges { node { name group state historyLineCount } }
        }
      }
    }
    """
    filt = json.dumps({"group": {"$regex": f"^{re.escape(group_prefix)}"}})
    cursor, best = None, None
    while True:
        data = _wandb_graphql(query, {"entity": entity, "project": project, "cursor": cursor, "regex": filt})
        runs = data["project"]["runs"]
        for edge in runs["edges"]:
            node = edge["node"]
            if best is None or (node["historyLineCount"] or 0) > (best["historyLineCount"] or 0):
                best = node
        if not runs["pageInfo"]["hasNextPage"]:
            break
        cursor = runs["pageInfo"]["endCursor"]
    if best is None or not best["historyLineCount"]:
        raise RuntimeError(f"No populated wandb runs found for group prefix '{group_prefix}'")
    return best


def fetch_train_loss_history(run_name, entity=WANDB_ENTITY, project=WANDB_PROJECT,
                              key=WANDB_LOSS_KEY, max_samples=10000):
    query = """
    query Q($entity:String!,$project:String!,$name:String!,$specs:[JSONString!]!){
      project(name:$project, entityName:$entity){
        run(name:$name){ sampledHistory(specs:$specs) }
      }
    }
    """
    spec = json.dumps({"keys": ["_step", key], "samples": max_samples})
    data = _wandb_graphql(query, {"entity": entity, "project": project, "name": run_name, "specs": [spec]})
    rows = data["project"]["run"]["sampledHistory"][0]
    loss_by_step = {int(r["_step"]): r[key] for r in rows if key in r}
    return {s: l for s, l in loss_by_step.items() if l == l}  # drop NaNs (l == l is False for NaN)


# Some models' training history can't be read off a single wandb run. pythia-1b's fp16 run
# ('v2-1b_1ifzt6l1') diverged to NaN loss around step ~127728, and EleutherAI restarted large
# stretches of training in bf16 afterwards (see 'v2-1b-bf16_*' groups). find_full_training_run()
# can't discover this automatically since no single run spans the full 0-143000 range, so we hand
# -curate the merge instead: earlier segments are the fp16 base, later ones are the bf16 fixes and
# take priority wherever they overlap.
WANDB_LOSS_SEGMENTS = {
    "EleutherAI/pythia-1b": [
        {"run": "1n8lyorh", "group": "v2-1b_1ifzt6l1"},       # fp16 base, steps 0-127728 (ends in NaN)
        {"run": "339s0ka6", "group": "v2-1b-bf16_dhizomr8"},  # bf16 restart, steps ~9158-104290
        {"run": "1ena81fo", "group": "v2-1b-bf16_14geqde9"},  # bf16 continuation, ~107145-119890
        {"run": "318z89xb", "group": "v2-1b-bf16_h2e543aw"},  # bf16 continuation, ~126145-138546
    ],
}


def fetch_train_loss_history_merged(segments, **kwargs):
    """Merges multiple wandb run segments into one {step: loss} map. Later segments override earlier
    ones for any overlapping step, so a bf16 restart that fixed an fp16 divergence wins over the run
    it superseded."""
    merged = {}
    for seg in segments:
        merged.update(fetch_train_loss_history(seg["run"], **kwargs))
    return merged


def get_loss_history_and_wandb_meta(model_name):
    if model_name in WANDB_LOSS_SEGMENTS:
        segments = WANDB_LOSS_SEGMENTS[model_name]
        print(f"Using {len(segments)} hand-curated wandb run segment(s) for {model_name} "
              "(training switched fp16->bf16 partway through)")
        loss_by_step = fetch_train_loss_history_merged(segments)
        val_loss_by_step = fetch_train_loss_history_merged(segments, key=WANDB_VAL_LOSS_KEY)
        wandb_meta = {"entity": WANDB_ENTITY, "project": WANDB_PROJECT, "segments": segments}
    else:
        print(f"Locating the full-length wandb training run for {model_name} ...")
        run_info = find_full_training_run(model_name)
        print(f"Using wandb run '{run_info['name']}' (group '{run_info['group']}', "
              f"{run_info['historyLineCount']} logged rows)")
        loss_by_step = fetch_train_loss_history(run_info["name"])
        val_loss_by_step = fetch_train_loss_history(run_info["name"], key=WANDB_VAL_LOSS_KEY)
        wandb_meta = {"entity": WANDB_ENTITY, "project": WANDB_PROJECT,
                       "run": run_info["name"], "group": run_info["group"]}
    print(f"Loaded {len(loss_by_step)} logged train-loss points spanning steps "
          f"{min(loss_by_step)}-{max(loss_by_step)}")
    # Validation/test loss is logged far more sparsely than train loss (EleutherAI only ran eval
    # occasionally), so we keep it as its own sparse {step: loss} series rather than interpolating
    # it onto every checkpoint step.
    if val_loss_by_step:
        print(f"Loaded {len(val_loss_by_step)} validation-loss point(s) at step(s) "
              f"{sorted(val_loss_by_step)}")
    else:
        print("No validation-loss points found for this run.")
    return loss_by_step, val_loss_by_step, wandb_meta


def nearest_loss(loss_by_step, step):
    if not loss_by_step:
        return None
    nearest = min(loss_by_step, key=lambda s: abs(s - step))
    return loss_by_step[nearest]


def measure_checkpoint(model_name, step, bit_depths):
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=f"step{step}", cache_dir=CACHE_DIR)
    num_param = sum(p.numel() for p in model.parameters()) / 1e6
    c_bin, c_multi, _ = measure_complexity(model, bit_depths=bit_depths)
    c_bit_comp = measure_bitplane_compression(model, bit_depths=bit_depths)
    del model
    return num_param, c_bin, c_multi, c_bit_comp


REQUIRED_HISTORY_KEYS = {"steps", "tokens", "train_loss", "sav_bin", "c_bin_raw",
                          "sav_multi", "sav_msb", "multi_per_plane", "sav_lzma", "sav_gzip"}


def _fresh_export(model_name, wandb_meta, val_loss_by_step=None):
    val_loss_by_step = val_loss_by_step or {}
    return {
        "metadata": {
            "model": model_name,
            "bit_depths": BIT_DEPTHS,
            "tokens_per_step": TOKENS_PER_STEP,
            "wandb": wandb_meta,
            # sparse {step: loss} series, kept separate from per-checkpoint history since eval
            # points don't align with our checkpoint schedule
            "val_loss": {"steps": sorted(val_loss_by_step),
                         "loss": [val_loss_by_step[s] for s in sorted(val_loss_by_step)]},
        },
        "history": {
            "steps": [], "tokens": [], "train_loss": [],
            "sav_bin": [], "c_bin_raw": [],  # c_bin_raw: absolute BDM score of the binarized weights
            "sav_multi": {str(bd): [] for bd in BIT_DEPTHS},
            "sav_msb": {str(bd): [] for bd in BIT_DEPTHS},  # savings for the uppermost (MSB) bitplane only
            "multi_per_plane": {str(bd): [] for bd in BIT_DEPTHS},  # raw per-plane BDM scores, LSB->MSB
            "sav_lzma": {str(bd): [] for bd in BIT_DEPTHS},
            "sav_gzip": {str(bd): [] for bd in BIT_DEPTHS},
        },
    }


def _load_resumable_export(results_path, model_name, wandb_meta, val_loss_by_step=None):
    """Reuses a prior results file so an interrupted/rerun sweep never recomputes a checkpoint that
    was already measured. Only resumes if the file matches this config AND already carries every raw
    field we need (older result files predate e.g. the per-plane MSB breakdown and must be redone)."""
    if not os.path.exists(results_path):
        return _fresh_export(model_name, wandb_meta, val_loss_by_step), set()
    with open(results_path) as f:
        existing = json.load(f)
    meta = existing.get("metadata", {})
    if meta.get("model") != model_name or meta.get("bit_depths") != BIT_DEPTHS:
        return _fresh_export(model_name, wandb_meta, val_loss_by_step), set()
    if not REQUIRED_HISTORY_KEYS.issubset(existing.get("history", {}).keys()):
        print("Existing results file is missing fields this version logs (e.g. per-plane MSB data); "
              "recomputing all checkpoints.")
        return _fresh_export(model_name, wandb_meta, val_loss_by_step), set()
    existing["metadata"]["wandb"] = wandb_meta
    val_loss_by_step = val_loss_by_step or {}
    existing["metadata"]["val_loss"] = {"steps": sorted(val_loss_by_step),
                                         "loss": [val_loss_by_step[s] for s in sorted(val_loss_by_step)]}
    done_steps = set(existing["history"]["steps"])
    print(f"Resuming from {results_path}: {len(done_steps)} checkpoint(s) already logged, skipping those.")
    return existing, done_steps


def main(model_name):
    loss_by_step, val_loss_by_step, wandb_meta = get_loss_history_and_wandb_meta(model_name)

    save_name = model_name.split("/")[-1]
    results_path = os.path.join(RESULTS_DIR, f"{save_name}_dynamics.json")

    export, done_steps = _load_resumable_export(results_path, model_name, wandb_meta, val_loss_by_step)
    h = export["history"]

    baseline = None
    if 0 in done_steps:
        idx0 = h["steps"].index(0)
        baseline = (h["c_bin_raw"][idx0], {bd: h["multi_per_plane"][str(bd)][idx0] for bd in BIT_DEPTHS})

    for step in TARGET_STEPS:
        if step in done_steps:
            continue
        t0 = time.time()
        num_param, c_bin, c_multi, c_bit_comp = measure_checkpoint(model_name, step, BIT_DEPTHS)
        export["metadata"]["num_param"] = num_param
        if baseline is None:
            baseline = (c_bin, c_multi)
        b_bin, b_multi = baseline

        h["steps"].append(step)
        h["tokens"].append(step * TOKENS_PER_STEP)
        h["train_loss"].append(nearest_loss(loss_by_step, step))
        h["sav_bin"].append((1 - c_bin / b_bin) * 100 if b_bin else 0.0)
        h["c_bin_raw"].append(c_bin)
        for bd in BIT_DEPTHS:
            b_sum = sum(b_multi[bd])
            h["sav_multi"][str(bd)].append((1 - sum(c_multi[bd]) / b_sum) * 100 if b_sum else 0.0)
            # c_multi[bd] is ordered LSB (index 0) -> MSB (index bd-1); see qbdm/qbdm.py measure_complexity
            b_msb, c_msb = b_multi[bd][-1], c_multi[bd][-1]
            h["sav_msb"][str(bd)].append((1 - c_msb / b_msb) * 100 if b_msb else 0.0)
            h["multi_per_plane"][str(bd)].append(list(c_multi[bd]))
            h["sav_lzma"][str(bd)].append(c_bit_comp[bd]["lzma"])
            h["sav_gzip"][str(bd)].append(c_bit_comp[bd]["gzip"])

        loss_val = h["train_loss"][-1]
        loss_str = f"{loss_val:.3f}" if loss_val is not None else "n/a"
        print(f"step {step:>7}: loss={loss_str}, BDM sav={h['sav_bin'][-1]:6.2f}%, "
              f"MSB(bit{BIT_DEPTHS[-1]-1}) sav={h['sav_msb'][str(BIT_DEPTHS[-1])][-1]:6.2f}%, "
              f"LZMA sav={h['sav_lzma'][str(BIT_DEPTHS[-1])][-1]:6.2f}% ({time.time() - t0:.1f}s)")

        with open(results_path, "w") as f:
            json.dump(export, f, indent=2)

    return results_path


if __name__ == "__main__":
    from utils.plotter import make_pythia_dynamics_figure
    for model_name in MODEL_NAMES:
        print(f"\n=== {model_name} ===")
        results_path = main(model_name)
        make_pythia_dynamics_figure(results_path, RESULTS_DIR)
