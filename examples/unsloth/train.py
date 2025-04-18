import os
from typing import Optional

from pydantic import BaseModel

from nebu import Message
from nebu.processors.decorate import processor
from nebu.processors.models import (
    V1Scale,
    V1ScaleDown,
    V1ScaleUp,
    V1ScaleZero,
)


class TrainingRequest(BaseModel):
    adapter_name: str
    dataset: str
    model: str = "unsloth/Qwen2.5-VL-32B-Instruct"
    max_length: int = 65536
    epochs: int = 5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 5
    lora_alpha: int = 128
    lora_rank: int = 64
    lora_dropout: float = 0
    optimizer: str = "adamw_8bit"
    owner: Optional[str] = None


class TrainingResponse(BaseModel):
    loss: float
    train_steps_per_second: float
    train_samples_per_second: float
    train_runtime: float
    adapter_name: str
    adapter_uri: str


# TODO: add default scale
scale = V1Scale(
    up=V1ScaleUp(above_pressure=10, duration="5m"),
    down=V1ScaleDown(below_pressure=2, duration="10m"),
    zero=V1ScaleZero(duration="10m"),
)

setup_script = """
apt update
apt install -y git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install trl peft transformers bitsandbytes sentencepiece accelerate
pip uninstall -y xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install -e git+https://github.com/pbarker/unsloth-zoo.git#egg=unsloth_zoo
pip install -e git+https://github.com/pbarker/unsloth.git#egg=unsloth
"""


@processor(
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    setup_script=setup_script,
    scale=scale,
    accelerators=["1:A100_SXM"],
    platform="runpod",
)
def train_unsloth_sft(message: Message[TrainingRequest]) -> TrainingResponse:
    import json
    import time

    import requests
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator

    from nebu import (
        Adapter,
        Bucket,
        Cache,
        ContainerConfig,
        find_latest_checkpoint,
        is_allowed,
        oai_to_unsloth,
    )

    print("message", message)
    training_request: TrainingRequest = message.content
    if not training_request:
        raise ValueError("No training request provided")

    container_config = ContainerConfig.from_env()
    print("container_config", container_config)

    cache = Cache()
    bucket = Bucket()

    print("loading model...")
    adapter_uri = f"{container_config.namespace_volume_uri}/adapters/{training_request.adapter_name}"
    time_start_load = time.time()
    model = None

    cache_key = f"adapters:{training_request.adapter_name}"
    print("checking cache for adapter", cache_key)
    val_raw = cache.get(cache_key)

    is_continue = False
    epochs_trained = 0
    if val_raw:
        adapter = Adapter.model_validate_json(val_raw)
        print("Found adapter: ", adapter)

        epochs_trained = adapter.epochs_trained

        if not is_allowed(adapter.owner, message.user_id, message.orgs):
            raise ValueError("You are not allowed to train this existing adapter")

        time_start = time.time()
        bucket.sync(adapter.uri, "/latest")
        print(f"Synced in {time.time() - time_start} seconds")

        model, tokenizer = FastVisionModel.from_pretrained(
            "/latest",
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
            max_seq_length=65_536,
            dtype=torch.bfloat16,
        )
        is_continue = True
    if not model:
        print("Loading model from scratch")
        model, tokenizer = FastVisionModel.from_pretrained(
            training_request.model,
            load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
            dtype=torch.bfloat16,
            max_seq_length=65_536,
        )

        print("getting peft model...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=True,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=training_request.lora_rank,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=training_request.lora_alpha,  # Recommended alpha == r at least
            lora_dropout=training_request.lora_dropout,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            use_fast=True,
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )
    print(f"Loaded model in {time.time() - time_start_load} seconds")

    print("Downloading dataset")
    time_start_download = time.time()
    response = requests.get(training_request.dataset)
    response.raise_for_status()  # optional: raises if request failed
    print(f"Downloaded dataset in {time.time() - time_start_download} seconds")

    # Decode and split into lines
    lines = response.content.decode("utf-8").splitlines()

    # Parse and convert each JSON line
    time_start_convert = time.time()
    converted_dataset = [
        oai_to_unsloth(json.loads(line)) for line in lines if line.strip()
    ]
    print(f"Converted dataset in {time.time() - time_start_convert} seconds")

    print(converted_dataset)

    FastVisionModel.for_training(model)  # Enable for training!

    train_epochs = epochs_trained + training_request.epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=training_request.batch_size,
            gradient_accumulation_steps=training_request.gradient_accumulation_steps,
            warmup_steps=training_request.warmup_steps,
            # max_steps=training_request.max_steps,
            num_train_epochs=train_epochs,
            learning_rate=training_request.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=training_request.logging_steps,
            optim=training_request.optimizer,
            weight_decay=training_request.weight_decay,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # For Weights and Biases
            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=training_request.max_length,
        ),
    )

    # Set environment variable for Unsloth to return logits
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    time_start_train = time.time()
    trainer_stats = trainer.train(resume_from_checkpoint=is_continue)
    print(trainer_stats)
    print(f"Trained in {time.time() - time_start_train} seconds")

    latest_checkpoint = find_latest_checkpoint("outputs")
    print("latest checkpoint")
    if latest_checkpoint:
        print("Copying latest checkpoint to bucket")
        bucket.copy(
            latest_checkpoint,
            adapter_uri,
        )

    # TODO: store this in the bucket so we don't need to copy every time
    adapter = Adapter(
        name=training_request.adapter_name,
        uri=adapter_uri,
        owner=message.content.owner if message.content.owner else message.user_id,  # type: ignore
        base_model=training_request.model,
        epochs_trained=train_epochs,
        last_trained=int(time.time()),
        lora_rank=training_request.lora_rank,
        lora_alpha=training_request.lora_alpha,
        lora_dropout=training_request.lora_dropout,
    )
    cache.set(cache_key, adapter.model_dump_json())

    return TrainingResponse(
        loss=trainer_stats.training_loss,
        train_steps_per_second=trainer_stats.metrics["train_steps_per_second"],
        train_samples_per_second=trainer_stats.metrics["train_samples_per_second"],
        train_runtime=trainer_stats.metrics["train_runtime"],
        adapter_name=training_request.adapter_name,
        adapter_uri=adapter_uri,
    )
