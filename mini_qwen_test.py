import numpy as np
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from layers import ImageEncoder, LoRALinear
from pathlib import Path


# ========== CONFIG ==========

MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "/Users/nnn/PycharmProjects/MLScratch/flickr8k_local"
CHECKPOINT_PATH = "mini_qwen_vl_ranking_best.pt"

RUN_NAME = Path(CHECKPOINT_PATH).stem
OUTPUT_DIR = Path("results") / "test" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMG_SIZE = 128
EMBED_DIM = 256
N_TEST = 300

dataset = load_from_disk(DATASET_PATH)

val_dataset = dataset["test"].select(
    range(min(N_TEST, len(dataset["test"])))
)

N_TEST = len(val_dataset)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ========== DATASET ==========

if N_TEST < 2:
    raise ValueError("Need at least 2 validation images.")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ========== TOKENIZER ==========

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True
)


# ========== BUILD MODELS ==========

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    local_files_only=True
).to(device)

img_encoder = ImageEncoder(EMBED_DIM).to(device)

projector = nn.Sequential(
    nn.Linear(EMBED_DIM, 768),
    nn.GELU(),
    nn.Linear(768, 1536)
).to(device)


# Same LoRA configuration as training
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(
        layer.self_attn.q_proj,
        r=8,
        alpha=16
    )
    layer.self_attn.v_proj = LoRALinear(
        layer.self_attn.v_proj,
        r=8,
        alpha=16
    )


# ========== LOAD CHECKPOINT ==========

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint validation loss:", checkpoint.get("val_loss"))

img_encoder.load_state_dict(
    checkpoint["image_encoder"]
)

projector.load_state_dict(
    checkpoint["projector"]
)

# Restore trained LoRA parameters
lora_state = checkpoint["lora"]
model_params = dict(model.named_parameters())

with torch.no_grad():
    for name, value in lora_state.items():
        if name not in model_params:
            raise KeyError(
                f"Missing LoRA parameter: {name}"
            )

        model_params[name].copy_(
            value.to(
                device=model_params[name].device,
                dtype=model_params[name].dtype
            )
        )

print("Checkpoint loaded!")
print("Epoch:", checkpoint["epoch"])


# ========== EVALUATION MODE ==========

model.eval()
img_encoder.eval()
projector.eval()


# ========== IMAGE EMBEDDING DIVERSITY TEST ==========

import torch.nn.functional as F

@torch.inference_mode()
def image_embedding(image):
    x = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    z = projector(img_encoder(x))  # (1, 64, 1536)

    return z.flatten(1).float()


embeddings = torch.cat([
    image_embedding(val_dataset[i]["image"])
    for i in range(N_TEST)
], dim=0)

embeddings = F.normalize(embeddings, dim=-1)

similarity = embeddings @ embeddings.T

mask = ~torch.eye(
    N_TEST,
    dtype=torch.bool,
    device=device
)

print("\n========== EMBEDDING DIVERSITY ==========")
print("Mean similarity:", similarity[mask].mean().item())
print("Min similarity:", similarity[mask].min().item())
print("Max similarity:", similarity[mask].max().item())
print("========================================")

# ========== PROMPT ==========

messages = [
    {
        "role": "user",
        "content": "Describe the image."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

prompt_ids = tokenizer(
    prompt,
    add_special_tokens=False
)["input_ids"]

# Match training's caption-ending convention
answer_end = tokenizer.eos_token


# ========== CAPTION LOSS ==========

@torch.inference_mode()
def caption_loss(image, caption, zero_image=False):

    # Image -> visual tokens
    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    img_tokens = projector(
        img_encoder(image_tensor)
    )  # (1, 64, 1536)

    if zero_image:
        img_tokens = torch.zeros_like(img_tokens)

    # Ground-truth caption tokens
    answer_ids = tokenizer(
        caption + answer_end,
        add_special_tokens=False
    )["input_ids"]

    input_ids = torch.tensor(
        [prompt_ids + answer_ids],
        dtype=torch.long,
        device=device
    )

    labels = torch.tensor(
        [[-100] * len(prompt_ids) + answer_ids],
        dtype=torch.long,
        device=device
    )

    text_tokens = model.model.embed_tokens(input_ids)

    img_tokens = img_tokens.to(text_tokens.dtype)

    combined = torch.cat(
        [img_tokens, text_tokens],
        dim=1
    )

    # Ignore visual tokens in the loss
    image_labels = torch.full(
        (1, img_tokens.shape[1]),
        -100,
        dtype=torch.long,
        device=device
    )

    combined_labels = torch.cat(
        [image_labels, labels],
        dim=1
    )

    attention_mask = torch.ones(
        combined.shape[:2],
        dtype=torch.long,
        device=device
    )

    output = model(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        use_cache=False
    )

    shift_logits = output.logits[:, :-1, :].float()
    shift_labels = combined_labels[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100
    )

    return loss.item()


# ========== VISUAL GROUNDING EVALUATION ==========

correct_losses = []
wrong_losses = []
zero_losses = []

print(f"\nTesting {N_TEST} validation images...\n")

rng = np.random.default_rng(42)
indices = np.arange(N_TEST)

while True:
    wrong_indices = rng.permutation(N_TEST)
    if np.all(wrong_indices != indices):
        break

for i in range(N_TEST):
    sample = val_dataset[i]

    wrong_sample = val_dataset[int(wrong_indices[i])]

    caption = sample["caption_0"]

    correct_losses.append(
        caption_loss(sample["image"],caption)
    )

    wrong_losses.append(
        caption_loss(wrong_sample["image"],caption)
    )

    zero_losses.append(
        caption_loss(sample["image"],caption,zero_image=True)
    )

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{N_TEST}")

correct = np.asarray(correct_losses)
wrong = np.asarray(wrong_losses)
zero = np.asarray(zero_losses)

wrong_gap = wrong - correct
zero_gap = zero - correct

np.savez(
    OUTPUT_DIR / "grounding_results.npz",
    correct=correct,
    wrong=wrong,
    zero=zero,
    wrong_gap=wrong_gap,
    zero_gap=zero_gap,
    wrong_indices=wrong_indices
)


# ========== RESULTS ==========

def bootstrap_ci(values, n_boot=10000):
    rng = np.random.default_rng(123)
    means = np.empty(n_boot)

    for b in range(n_boot):
        sample = rng.choice(
            values, size=len(values), replace=True
        )

        means[b] = sample.mean()

    return np.percentile(means, [2.5, 97.5])

ci_wrong = bootstrap_ci(wrong_gap)
ci_zero = bootstrap_ci(zero_gap)

print("\n========== FINAL EVALUATION ==========")
print(f"Samples:              {N_TEST}")
print(f"Correct avg loss:     {correct.mean():.5f}")
print(f"Wrong avg loss:       {wrong.mean():.5f}")
print(f"Zero avg loss:        {zero.mean():.5f}")

print(f"\nWrong gap:          {wrong_gap.mean():+.5f}")
print(f"Wrong gap std:        {wrong_gap.std(ddof=1):.5f}")
print(f"Wrong gap 95% CI:     {ci_wrong}")
print(f"Correct-image wins:   {(wrong_gap > 0).mean():.1%}")

print(f"\nZero gap:           {zero_gap.mean():+.5f}")
print(f"Zero gap 95% CI:      {ci_zero}")

print("======================================")
import matplotlib.pyplot as plt


FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.hist(wrong_gap, bins=25, alpha=0.75)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Wrong-image loss minus correct-image loss")
plt.ylabel("Number of samples")
plt.title("Visual Grounding: Paired Loss Differences")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "grounding_loss_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




