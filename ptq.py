import json
import torch
import timm
from datasets import load_dataset
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
import copy

from ptq_utils import UniformQuantizer, UniformQuantizer_per_channel, attach_weight_quantizers, toggle_quantization

BIT = [1, 2, 3, 4, 5, 6, 7, 8]
EXCLUDE = []
BATCH_SIZE = 256

MODELS = [
    "resnet18",
    "resnet50",
    "vit_base_patch16_224",
    "efficientnet_b0",
    "mobilenetv3_large_100",
]

HF_TOKEN = ""

def get_dataset():
    return load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=HF_TOKEN)

def evaluate(model, dataset, device, dtype=torch.float32, desc="Evaluating", transform=None):
    model.eval()
    correct = 0
    total = 0
    batch_imgs, batch_labels = [], []

    with torch.no_grad():
        for sample in tqdm(dataset, desc=desc, total=50000):
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            batch_imgs.append(transform(img))
            batch_labels.append(sample["label"])

            if len(batch_imgs) == BATCH_SIZE:
                imgs = torch.stack(batch_imgs).to(device=device, dtype=dtype)
                labels = torch.tensor(batch_labels).to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                batch_imgs, batch_labels = [], []

        if batch_imgs:
            imgs = torch.stack(batch_imgs).to(device=device, dtype=dtype)
            labels = torch.tensor(batch_labels).to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    results = {"bit_depths": BIT, "models": {}}

    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        results["models"][model_name] = {}

        try:
            # ── 1. Load pretrained model ──────────────────────────────────────
            model_f32 = timm.create_model(model_name, pretrained=True).eval().to(device)

            # Build transform once per model
            data_config = resolve_data_config({}, model=model_f32)
            transform = create_transform(**data_config)

            # ── 2. Full precision (FP32) ──────────────────────────────────────
            acc_fp32 = evaluate(model_f32, get_dataset(), device,
                                dtype=torch.float32, desc=f"{model_name} | FP32", transform=transform)
            results["models"][model_name]["fp32"] = acc_fp32
            print(f"  FP32  acc: {acc_fp32:.4f}")

            # ── 3. Half precision (FP16) ──────────────────────────────────────
            model_fp16 = copy.deepcopy(model_f32).half()
            acc_fp16 = evaluate(model_fp16, get_dataset(), device,
                                dtype=torch.float16, desc=f"{model_name} | FP16", transform=transform)
            results["models"][model_name]["fp16"] = acc_fp16
            print(f"  FP16  acc: {acc_fp16:.4f}")
            del model_fp16

            # ── 4. PTQ ────────────────────────────────────────────────────────
            results["models"][model_name]["ptq"] = {}
            for bit in BIT:
                model_q = copy.deepcopy(model_f32).to(device)
                attach_weight_quantizers(
                    model=model_q,
                    exclude_layers=EXCLUDE,
                    quantizer=UniformQuantizer_per_channel(bit_width=bit),
                    enabled=False,
                    verbose=False,
                )
                toggle_quantization(model_q, enabled=True)
                acc_q = evaluate(model_q, get_dataset(), device,
                                 dtype=torch.float32, desc=f"{model_name} | {bit}-bit", transform=transform)
                results["models"][model_name]["ptq"][str(bit)] = acc_q
                print(f"  {bit}-bit acc: {acc_q:.4f}")
                del model_q

            del model_f32
            print(f"  Done.")

        except Exception as e:
            import traceback
            print(f"  Error processing {model_name}: {e}")
            traceback.print_exc()

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), f"ptq.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")