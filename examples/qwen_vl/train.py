import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch

# Add PEFT imports
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
)
from PIL import Image
from pydantic import BaseModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from nebu import (
    Adapter,
    Bucket,
    Cache,
    ContainerConfig,
    Message,
    is_allowed,
    processor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_image(url: str) -> Optional[Image.Image]:
    """Downloads an image from a URL and returns a PIL Image or None on failure."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to process image from {url}: {e}")
        return None


class OpenAIChatDataset(Dataset):
    """Dataset for loading OpenAI chat format from a JSONL file."""

    def __init__(self, jsonl_path: str):
        self.data = []
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {jsonl_path}")
        except FileNotFoundError:
            logger.error(f"JSONL file not found at {jsonl_path}")
            raise
        logger.info(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        # Returns the raw message structure
        return self.data[idx]


def load_model_and_processor(
    model_name_or_path: str,  # Can be base model name or path to adapter checkpoint
    torch_dtype_str: str,
    attn_implementation: Optional[str],
    train_request: "TrainingRequest",  # Pass the request
    existing_adapter: Optional[
        Adapter
    ] = None,  # Pass existing adapter metadata if found
) -> Tuple[Union[PreTrainedModel, PeftModel, PeftMixedModel], PreTrainedTokenizerBase]:
    """Loads the model and processor, applying PEFT/LoRA if configured."""
    logger.info(f"Loading model and processor for: {model_name_or_path}")

    # Determine the base model name
    # If an existing adapter is provided, use its base model, otherwise use the request's model
    base_model_name = (
        existing_adapter.base_model if existing_adapter else train_request.model
    )
    logger.info(f"Using base model name: {base_model_name}")

    # Map string representation to torch.dtype
    dtype_map = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(torch_dtype_str.lower())
    if torch_dtype is None:
        logger.warning(
            f"Invalid torch_dtype string: '{torch_dtype_str}'. Falling back to torch.bfloat16."
        )
        torch_dtype = torch.bfloat16

    attn_impl = attn_implementation if attn_implementation else None
    logger.info(f"Using torch_dtype: {torch_dtype}, attn_implementation: {attn_impl}")

    # Load the base model
    logger.info(f"Loading base model weights: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,  # Always load base weights from the original name
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,  # Needed for some models like Qwen-VL
    )
    # Load the processor using the same base model name
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")

    if attn_implementation == "flash_attention_2":
        tokenizer.padding_side = "left"
        logger.info("Set tokenizer padding_side to left for flash_attention_2")

    # --- PEFT/LoRA Handling (Assuming always enabled) ---
    # Check if model_name_or_path points to a downloaded adapter checkpoint
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    # We also need existing_adapter metadata to confirm it's compatible
    if os.path.exists(adapter_config_path) and existing_adapter:
        logger.info(f"Loading PEFT adapter from checkpoint: {model_name_or_path}")
        # Load the PEFT model from the checkpoint, attaching it to the base model
        try:
            # The `model` variable here is the base_model loaded earlier
            loaded_peft_model = PeftModel.from_pretrained(
                model,  # Pass the loaded base model
                model_name_or_path,
                is_trainable=True,  # Ensure adapter weights are trainable
            )
            model = loaded_peft_model  # Assign to model only on success
            logger.info("Successfully loaded PEFT adapter.")
        except Exception as e:
            logger.error(f"Failed to load PEFT adapter from {model_name_or_path}: {e}")
            logger.warning("Falling back to creating a new PEFT config.")
            # Fallback: Apply new config to the ORIGINAL base model
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # QwenVL is causal LM
                r=train_request.lora_rank,
                lora_alpha=train_request.lora_alpha,
                lora_dropout=train_request.lora_dropout,
                target_modules=train_request.lora_target_modules,
                bias="none",
            )
            # Apply to the original base model stored in `model` before the try block
            model = get_peft_model(model, peft_config)
            logger.info("Applied new PEFT config after adapter load failure.")
    else:
        # If no existing adapter checkpoint found, create and apply a new LoRA config
        logger.info(
            "No valid PEFT adapter checkpoint found or provided. Applying new PEFT LoRA configuration."
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # QwenVL is causal LM
            r=train_request.lora_rank,
            lora_alpha=train_request.lora_alpha,
            lora_dropout=train_request.lora_dropout,
            target_modules=train_request.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info("Applied new PEFT config.")

    # Print trainable parameters for verification
    model.print_trainable_parameters()

    logger.info("Model and processor loading complete.")
    return model, processor


def collate_fn(batch: List[Dict[str, Any]], processor: Any) -> Dict[str, torch.Tensor]:
    """Prepares a batch of data for the model."""
    processed_messages_list = []
    image_list = []

    for item in batch:
        messages = item.get("messages", [])
        current_images = []
        processed_content = []

        # Process message content, separating text and images
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not role or not content:
                continue

            if isinstance(content, str):  # Simple text content
                processed_content.append({"role": role, "content": content})
            elif isinstance(
                content, list
            ):  # List content (potentially text and images)
                message_parts = []
                for part in content:
                    if part.get("type") == "text":
                        message_parts.append(
                            {"type": "text", "text": part.get("text", "")}
                        )
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url")
                        if img_url:
                            img = download_image(img_url)
                            if img_url and not img:
                                logger.warning(
                                    f"Skipping message part due to failed image download: {img_url}"
                                )
                                # Decide how to handle missing images (skip part, replace with text, etc.)
                                continue  # Skip this image part
                            elif img:
                                current_images.append(img)
                                message_parts.append(
                                    {"type": "image"}
                                )  # Placeholder or specific token

                if message_parts:
                    processed_content.append({"role": role, "content": message_parts})

        if not processed_content:
            logger.warning(f"Skipping item {item} due to no processable content.")
            continue  # Skip if no valid content

        # Only add images if there was content processed for this item
        if processed_content:
            processed_messages_list.append(processed_content)
            image_list.append(
                current_images if current_images else None
            )  # Use None if no images
        else:
            # Handle case where an item resulted in no content (e.g., only failed image URLs)
            # Depending on strategy, you might add a placeholder or skip entirely.
            # Currently skipped by the 'continue' above.
            pass

    # Filter out empty messages before proceeding
    valid_indices = [i for i, msg in enumerate(processed_messages_list) if msg]
    if not valid_indices:
        logger.warning("Collate function resulted in an empty batch.")
        # Return an empty dictionary or raise an error, depending on desired behavior
        return {}

    processed_messages_list = [processed_messages_list[i] for i in valid_indices]
    image_list = [image_list[i] for i in valid_indices]

    # Apply chat template and tokenize
    # Note: add_generation_prompt=False because we provide the full conversation including the assistant's response for training.
    tokenizer = getattr(processor, "tokenizer", processor)
    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in processed_messages_list
    ]

    # Process images - this assumes a processor like Qwen-VL's
    # Adapt this based on your specific model/processor if it handles images differently
    try:
        # Attempt multimodal processing
        inputs = processor(
            text=texts, images=image_list, return_tensors="pt", padding=True
        )
    except Exception as e:
        raise RuntimeError("Failed to process images") from e

    # Create labels by shifting input_ids
    input_ids_tensor = inputs.get("input_ids")
    if not isinstance(input_ids_tensor, torch.Tensor):
        logger.error("input_ids not found or not a tensor in processor output.")
        return {}

    labels = input_ids_tensor.clone()

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    else:
        logger.warning(
            "pad_token_id not found on tokenizer. Padding tokens may not be ignored correctly."
        )

    # Ignore image-related tokens in labels if they exist (check processor/tokenizer specifics)
    # Example for Qwen-VL, adjust token IDs if necessary
    image_token_ids = getattr(processor, "image_token_ids", [])
    if not image_token_ids and hasattr(tokenizer, "convert_tokens_to_ids"):
        # Use the correct tokens for Qwen2.5-VL
        image_tokens_to_check = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        image_token_ids = [
            tok_id
            for tok in image_tokens_to_check  # Use the updated list here
            if (tok_id := tokenizer.convert_tokens_to_ids(tok))
            != tokenizer.unk_token_id  # Assuming unk_token_id is not None or handled
        ]
    logger.info(f"Identified potential image token IDs: {image_token_ids}")

    for img_tok_id in image_token_ids:
        # Ensure token exists (convert_tokens_to_ids might return unk_token_id)
        if img_tok_id != getattr(tokenizer, "unk_token_id", None):
            labels[labels == img_tok_id] = -100

    inputs["labels"] = labels
    return inputs


def train(
    model: Union[PreTrainedModel, PeftModel, PeftMixedModel],
    processor: Any,
    train_dataloader: DataLoader,
    output_dir: str,  # Added output_dir parameter
    learning_rate: float,  # Added
    epochs: int,  # Added
    max_grad_norm: float,  # Added
) -> Dict[str, float]:
    """Simplified training loop."""
    logger.info("Initializing training...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Use passed LR
    num_training_steps = epochs * len(train_dataloader)  # Use passed epochs
    logger.info(f"Total training steps: {num_training_steps}")
    model.train()
    global_step = 0
    final_avg_loss = 0.0
    total_epochs_trained = 0  # Keep track of epochs trained in this run

    for epoch in range(epochs):  # Use passed epochs
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}",
            position=0,
            leave=True,
        )

        for batch in train_dataloader:
            # Move batch to device
            batch = {
                k: v.to(model.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )  # Use passed norm

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            progress_bar.update()

        progress_bar.close()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
        total_epochs_trained += 1
        if epoch == epochs - 1:  # Store loss of the last epoch
            final_avg_loss = avg_epoch_loss

    logger.info(f"Training completed! Trained {total_epochs_trained} epochs.")

    # Save model
    logger.info(f"Saving model to {output_dir}...")  # Use output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Model and processor saved.")

    return {"final_average_loss": final_avg_loss}


# TODO: Make setup script more specific if needed
setup_script = """
pip install -q -U transformers datasets torch Pillow requests pydantic tqdm accelerate sentencepiece nebu peft
pip install flash-attn --no-build-isolation
"""


class TrainingRequest(BaseModel):
    model: str
    dataset: str
    adapter_name: str
    owner: Optional[str] = None
    learning_rate: float = 5e-5
    epochs: int = 1
    batch_size: int = 1
    max_grad_norm: float = 1.0
    torch_dtype: str = "torch.bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]


class TrainingResponse(BaseModel):
    loss: float
    runtime: float
    adapter_name: str
    adapter_uri: str


@processor(
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    setup_script=setup_script,
    platform="runpod",
    accelerators=["1:A100_SXM"],
)
def train_qwen_vl(message: Message[TrainingRequest]) -> TrainingResponse:
    start_time = time.time()

    train_request = message.content
    if not train_request:
        msg = "No training request provided. Exiting."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Received training request: {train_request}")

    # Initialize Nebu components
    try:
        container_config = ContainerConfig.from_env()
        logger.info(f"Container config: {container_config}")
        cache = Cache()
        bucket = Bucket()
    except Exception as e:
        logger.error(f"Failed to initialize Nebu components: {e}")
        raise RuntimeError("Failed to initialize Nebu components") from e

    # Define paths
    adapter_uri = (
        f"{container_config.namespace_volume_uri}/adapters/{train_request.adapter_name}"
    )
    local_model_dir = "/tmp/model_checkpoint"  # Local dir for loading/saving
    os.makedirs(local_model_dir, exist_ok=True)

    # Check cache for existing adapter
    model_load_path = train_request.model  # Default to base model name
    epochs_trained = 0
    adapter_owner = train_request.owner if train_request.owner else message.user_id
    if not adapter_owner:
        msg = "Adapter owner could not be determined (neither provided in request nor available from message context)."
        logger.error(msg)
        raise ValueError(msg)

    cache_key = f"adapters:{train_request.adapter_name}"
    logger.info(f"Checking cache for adapter: {cache_key}")
    val_raw = cache.get(cache_key)
    existing_adapter_metadata: Optional[Adapter] = None

    if val_raw:
        try:
            adapter = Adapter.model_validate_json(val_raw)
            logger.info(f"Found existing adapter in cache: {adapter}")
            existing_adapter_metadata = adapter  # Store it

            # Validate owner
            if not is_allowed(adapter.owner, message.user_id, message.orgs):
                msg = f"User {message.user_id} not allowed to train existing adapter {train_request.adapter_name} owned by {adapter.owner}"
                logger.error(msg)
                raise PermissionError(msg)

            # Compatibility Check (Assuming PEFT is always used and adapter.lora_rank is never None)
            # If we reach here, the adapter exists and we assume it was trained with LoRA.
            logger.info(
                f"Adapter '{adapter.name}' found and assumed compatible with PEFT. Base model: {adapter.base_model}, Rank: {adapter.lora_rank}"
            )
            # Check if base model matches current request's base model
            if adapter.base_model != train_request.model:
                logger.warning(
                    f"Adapter base model '{adapter.base_model}' differs from request base model '{train_request.model}'. "
                    f"Will load adapter onto '{adapter.base_model}'."
                )
                # Note: load_model_and_processor will use adapter.base_model

            # Sync the adapter since it's considered valid
            epochs_trained = adapter.epochs_trained
            logger.info(f"Syncing adapter from {adapter.uri} to {local_model_dir}...")
            sync_start_time = time.time()
            try:
                bucket.sync(adapter.uri, local_model_dir)
                logger.info(
                    f"Synced adapter in {time.time() - sync_start_time:.2f} seconds."
                )
                model_load_path = (
                    local_model_dir  # Point to the *synced adapter* directory
                )
            except Exception as sync_error:
                logger.error(f"Failed to sync adapter from {adapter.uri}: {sync_error}")
                logger.warning("Proceeding without using synced adapter checkpoint.")
                existing_adapter_metadata = None
                model_load_path = train_request.model  # Reset load path
                epochs_trained = 0  # Reset epochs

        except Exception as e:
            logger.warning(
                f"Failed to load or validate adapter from cache: {e}. Proceeding as if no adapter exists."
            )
            existing_adapter_metadata = None  # Reset on error
            model_load_path = train_request.model
            epochs_trained = 0
    else:
        logger.info("No existing adapter found in cache. Training from base model.")

    # Load model and processor using values from request and potentially existing adapter info
    logger.info(f"Attempting to load model/processor. Load path: {model_load_path}")
    try:
        model, processor = load_model_and_processor(
            model_name_or_path=model_load_path,  # Path to base model or synced adapter
            torch_dtype_str=train_request.torch_dtype,
            attn_implementation=train_request.attn_implementation,
            train_request=train_request,  # Pass the full request for PEFT config
            existing_adapter=existing_adapter_metadata,  # Pass adapter metadata if found and valid
        )
    except Exception as e:
        logger.error(f"Failed to load model/processor: {e}", exc_info=True)
        raise RuntimeError("Failed to load model/processor") from e

    logger.info(f"Downloading dataset from: {train_request.dataset}")
    # Assuming dataset is a URL pointing to a JSONL file saved locally
    dataset_path = "/tmp/dataset.jsonl"
    try:
        response = requests.get(
            train_request.dataset, stream=True, timeout=60
        )  # Increased timeout
        response.raise_for_status()
        with open(dataset_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Dataset downloaded and saved to {dataset_path}")
    except requests.exceptions.RequestException as e:
        msg = f"Failed to download dataset from {train_request.dataset}: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # Load data
    try:
        train_dataset = OpenAIChatDataset(dataset_path)
        if len(train_dataset) == 0:
            msg = "Training dataset is empty after loading. Exiting."
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Loaded {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise RuntimeError("Failed to load dataset") from e

    logger.info("Creating dataloader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_request.batch_size,  # Use batch_size from request
        collate_fn=lambda batch: collate_fn(batch, processor),  # Pass processor
        shuffle=True,
    )

    # --- Training ---
    logger.info("Starting training process...")
    metrics = train(
        model,
        processor,
        train_dataloader,
        output_dir=local_model_dir,
        learning_rate=train_request.learning_rate,
        epochs=train_request.epochs,
        max_grad_norm=train_request.max_grad_norm,
    )
    final_loss = metrics.get("final_average_loss", float("nan"))  # Use NaN if not found
    logger.info(f"Training finished. Final average loss: {final_loss:.4f}")

    # --- Save results to bucket and cache ---
    logger.info(
        f"Copying trained model from {local_model_dir} to bucket at {adapter_uri}..."
    )
    try:
        copy_start_time = time.time()
        bucket.copy(local_model_dir, adapter_uri)
        logger.info(
            f"Copied model to bucket in {time.time() - copy_start_time:.2f} seconds."
        )
    except Exception as e:
        logger.error(f"Failed to copy model to bucket: {e}")
        raise RuntimeError("Failed to save model checkpoint to bucket") from e

    # Update adapter metadata
    total_epochs_trained = (
        epochs_trained + train_request.epochs
    )  # Use epochs from request
    logger.info(
        f"Updating adapter metadata. Total epochs trained: {total_epochs_trained}"
    )
    adapter_metadata = Adapter(
        name=train_request.adapter_name,
        uri=adapter_uri,
        owner=adapter_owner,
        base_model=train_request.model,
        epochs_trained=total_epochs_trained,
        last_trained=int(time.time()),
        lora_rank=train_request.lora_rank,
        lora_alpha=train_request.lora_alpha,
        lora_dropout=train_request.lora_dropout,
        lora_target_modules=train_request.lora_target_modules,
        learning_rate=train_request.learning_rate,
    )

    try:
        cache.set(cache_key, adapter_metadata.model_dump_json())
        logger.info(f"Updated adapter metadata in cache: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to update cache for {cache_key}: {e}")
        # Non-fatal, but log it

    runtime = time.time() - start_time
    logger.info(f"Total execution time: {runtime:.2f} seconds")

    return TrainingResponse(
        loss=final_loss,
        runtime=runtime,
        adapter_name=train_request.adapter_name,
        adapter_uri=adapter_uri,
    )


if __name__ == "__main__":
    logging.info("Launching training job...")
    training_request = TrainingRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",  # Or a smaller model for faster local testing
        dataset="https://storage.googleapis.com/orign/testdata/nebu/clinton.jsonl",
        adapter_name="bar1",
    )
    print("Training request: ", training_request.model_dump_json(indent=2))

    train_qwen_vl(training_request)
    print("Training job completed.")
