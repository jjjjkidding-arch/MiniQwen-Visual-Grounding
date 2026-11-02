from torch import nn
from torchvision import transforms
from layers import ImageEncoder, LoRALinear
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import time
import torch.nn.functional as F

def encode_example(caption, tokenizer):
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

    answer = caption + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    answer_ids = tokenizer(
        answer, add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return input_ids, labels


#------------
#------------
dataset = load_from_disk(
    "/Users/nnn/PycharmProjects/MLScratch/flickr8k_local"
)
MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = "mini_qwen_vl.pt"

print("Done")

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/nnn/Qwen2.5-1.5B-Instruct",
    local_files_only=True
)

EPOCHS = 1
BATCH_SIZE = 2

LR_VISION = 1e-4
LR_PROJECTOR = 2e-4
LR_LORA = 2e-5

PATIENCE = 4

N_TRAIN = 6000
N_EVAL = 1000
IMG_SIZE = 128
EMBED_DIM = 256

train_dataset = dataset["train"].select(
    range(min(N_TRAIN, len(dataset["train"])))
)

VAL_dataset = dataset["dev"].select(
    range(min(N_EVAL, len(dataset["dev"])))
)
print("Train images:", len(train_dataset))
print("Validation images:", len(VAL_dataset))
print("Columns:", train_dataset.column_names)





projector = nn.Sequential(
    nn.Linear(EMBED_DIM, 768),
    nn.GELU(),
    nn.Linear(768, 1536)
).to(device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


#---------------------
#-----model loading----

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    local_files_only=True
).to(device)
print("Model ready!")

img_encoder = ImageEncoder(EMBED_DIM).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)
img_encoder.load_state_dict(
    checkpoint["image_encoder"],
    strict=True
)

print("Image encoder loaded successfully!")

for p in model.parameters():
    p.requires_grad = False

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

# Restore trained projector
projector.load_state_dict(checkpoint["projector"])

# Restore trained Qwen LoRA weights
model_params = dict(model.named_parameters())

with torch.no_grad():
    for name, value in checkpoint["lora"].items():
        if name not in model_params:
            raise KeyError(f"Missing parameter: {name}")

        assert model_params[name].shape == value.shape

        model_params[name].copy_(
            value.to(
                device=model_params[name].device,
                dtype=model_params[name].dtype
            )
        )

expected_lora = {
    name
    for name, p in model.named_parameters()
    if p.requires_grad
}

loaded_lora = set(checkpoint["lora"])

assert expected_lora == loaded_lora, (
    f"Missing: {expected_lora - loaded_lora}\n"
    f"Unexpected: {loaded_lora - expected_lora}"
)

print("Pretrained CNN, projector and LoRA loaded!")


trainable_lora = [
    (name, p)
    for name, p in model.named_parameters()
    if p.requires_grad
]

print("Trainable LoRA tensors:", len(trainable_lora))
print(
    "Trainable LoRA parameters:",
    sum(p.numel() for _, p in trainable_lora)
)

optimizer = torch.optim.AdamW([
    {
        "params": [p for p in model.parameters()
                   if p.requires_grad],
        "lr": LR_LORA
    },
    {
        "params": projector.parameters(),
        "lr": LR_PROJECTOR
    },
    {
        "params": img_encoder.parameters(),
        "lr": LR_VISION
    }
], weight_decay=0.01)

@torch.no_grad()
def validate():
    img_encoder.eval()
    projector.eval()
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(VAL_dataset), BATCH_SIZE):
        batch = VAL_dataset[
            start:start + BATCH_SIZE
        ]

        images = torch.stack([
            transform(img.convert("RGB"))
            for img in batch["image"]
        ]).to(device)

        captions = batch["caption_0"]

        encoded = [
            encode_example(caption, tokenizer)
            for caption in captions
        ]

        input_ids_list = [
            torch.tensor(x[0], dtype=torch.long)
            for x in encoded
        ]

        labels_list = [
            torch.tensor(x[1], dtype=torch.long)
            for x in encoded
        ]

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        ).to(device)

        img_tokens = projector(img_encoder(images))
        input_tokens = model.model.embed_tokens(input_ids)

        img_tokens = img_tokens.to(dtype=input_tokens.dtype)

        combined = torch.cat(
            [img_tokens, input_tokens], dim=1
        )

        image_labels = torch.full(
            (images.shape[0], img_tokens.shape[1]),
            -100,
            dtype=torch.long,
            device=device
        )

        combined_labels = torch.cat(
            [image_labels, labels], dim=1
        )

        input_mask = torch.zeros_like(input_ids)

        for i, ids in enumerate(input_ids_list):
            input_mask[i, :len(ids)] = 1

        image_mask = torch.ones(
            (images.shape[0], img_tokens.shape[1]),
            dtype=torch.long,
            device=device
        )

        combined_mask = torch.cat(
            [image_mask, input_mask], dim=1
        )

        output = model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False
        )

        n_tokens = (combined_labels != -100).sum().item()

        total_loss += output.loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens

def per_ex_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()

    B, T, V = shift_logits.shape

    token_losses = F.cross_entropy(
        shift_logits.float().reshape(-1, V),
        labels.reshape(-1),
        ignore_index = -100,
        reduction="none",
    ).reshape(B, T)

    valid = (labels != -100)

    return token_losses.sum(dim=1) / valid.sum(dim=1).clamp(min=1)




#------training-----
best_val_loss = float("inf")
bad_epochs = 0
margin = 0.2
lambda_rank = 0.5

for step in range(EPOCHS):
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    img_encoder.train()
    projector.train()
    model.train()

    indices = torch.randperm(len(train_dataset))
    epoch_loss = 0.0
    epoch_tokens = 0

    for start in range(0, len(train_dataset), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        images, captions = [], []

        optimizer.zero_grad()

        for idx in batch_indices:
            sample = train_dataset[int(idx)]

            image = transform(sample["image"].convert("RGB"))
            caption = sample[f"caption_{random.randint(0, 4)}"]

            images.append(image)
            captions.append(caption)


        images = torch.stack(images).to(device)

        encoded = [
            encode_example(caption, tokenizer) for caption in captions
        ]

        input_ids_list = [
            torch.tensor(item[0], dtype=torch.long) for item in encoded
        ]

        labels_list = [
            torch.tensor(item[1], dtype=torch.long) for item in encoded
        ]

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        ).to(device)#B,T,


        # forward
        img_tokens = projector(img_encoder(images)) #B,64,256--->B,64,1536

        B = images.shape[0]

        if B < 2:
            continue

        shift = torch.randint(1, B, (1,)).item()
        wrong_img_tokens = img_tokens.roll(shifts=shift, dims=0)

        input_tokens = model.model.embed_tokens(input_ids)#B,T,1536

        img_tokens = img_tokens.to(dtype=input_tokens.dtype)
        wrong_img_tokens = wrong_img_tokens.to(dtype=input_tokens.dtype)
        combined = torch.cat([img_tokens, input_tokens], dim=1)#B,T+64,1536
        combined_wrong = torch.cat([wrong_img_tokens, input_tokens], dim=1)

        image_labels = torch.full(
            (images.shape[0],img_tokens.shape[1]),-100,
            dtype=torch.long,
            device=device
        )#B,64
        combined_labels = torch.cat(
            [image_labels, labels], dim=1
        )#B,64+T

        #input_mask = (input_ids != tokenizer.pad_token_id).long() shape---===B,T
        input_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(input_ids_list):
            input_mask[i, :len(ids)] = 1

        image_mask = torch.ones(
            (images.shape[0],img_tokens.shape[1]),
            dtype=torch.long,
            device=device
        )
        combined_mask = torch.cat(
            [image_mask, input_mask], dim=1
        )#B,64+T

        #backward
        out_correct = model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False
        )

        out_wrong = model(
            inputs_embeds=combined_wrong,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False
        )
        shift_labels = combined_labels[:, 1:]
        loss_correct_each = per_ex_loss(out_correct.logits, shift_labels)
        loss_wrong_each = per_ex_loss(out_wrong.logits, shift_labels)

        if (
                not torch.isfinite(loss_correct_each).all().item()
                or not torch.isfinite(loss_wrong_each).all().item()
        ):
            raise RuntimeError(f"Non-finite loss")

        ranking_loss = F.relu(margin + loss_correct_each - loss_wrong_each).mean()
        loss_correct = loss_correct_each.mean()
        loss = loss_correct + lambda_rank * ranking_loss
        loss.backward()

        n_tokens = (combined_labels != -100).sum().item()
        epoch_loss += loss.item() * n_tokens
        epoch_tokens += n_tokens

        optimizer.step()

        batch_num = start // BATCH_SIZE + 1
        total_batches = (len(train_dataset) + BATCH_SIZE - 1) // BATCH_SIZE

        if batch_num % 100 == 0:
            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start_time
            seconds_per_batch = elapsed / batch_num
            eta_minutes = (
                    (total_batches - batch_num)
                    * seconds_per_batch / 60
            )

            print(
                f"Batch {batch_num}/{total_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_minutes:.1f} min",
                flush=True
            )


    val_loss = validate()

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time

    print(
        f"Epoch {step + 1}/{EPOCHS} | "
        f"Train: {epoch_loss / epoch_tokens:.4f} | "
        f"Val: {val_loss:.4f} | "
        f"Time: {elapsed:.2f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save({
            "image_encoder": {
                k: v.detach().cpu()
                for k, v in img_encoder.state_dict().items()
            },
            "projector": {
                k: v.detach().cpu()
                for k, v in projector.state_dict().items()
            },
            "lora": {
                name: p.detach().cpu()
                for name, p in model.named_parameters()
                if p.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "epoch": step + 1,
            "val_loss": val_loss
        }, "mini_qwen_vl_ranking_best.pt")

        print("New best checkpoint saved!")

    else:
        bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping triggered")
            break










