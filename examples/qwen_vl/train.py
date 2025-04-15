import gc
import json
import logging
import math
import os
import shutil
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

# Add bitsandbytes import for 8bit Adam
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
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
)

from nebu import (
    Adapter,
    Bucket,
    Cache,
    ContainerConfig,
    Message,
    is_allowed,
    oai_to_qwen,
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

    # Log the resolved dtype
    logger.info(f"Resolved torch_dtype for model loading: {torch_dtype}")

    attn_impl = attn_implementation if attn_implementation else None
    logger.info(f"Using attn_implementation: {attn_impl}")

    # --- PEFT/LoRA Handling --- Check if loading an existing adapter
    loading_existing_adapter = False
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    if os.path.exists(adapter_config_path) and existing_adapter:
        loading_existing_adapter = True
        logger.info(
            f"Planning to load existing PEFT adapter from: {model_name_or_path}"
        )

    # Use device_map="auto" for initial base model loading
    device_map = "auto"
    logger.info(f"Setting device_map='{device_map}' for base model load.")

    # Load the base model
    logger.info(f"Loading base model weights: {base_model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,  # Always load base weights from the original name
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map=device_map,  # Use determined device_map
        trust_remote_code=True,  # Needed for some models like Qwen-VL
    )
    # Load the processor using the same base model name
    processor = AutoProcessor.from_pretrained(
        base_model_name, trust_remote_code=True, use_fast=True
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")

    if attn_implementation == "flash_attention_2":
        tokenizer.padding_side = "left"
        logger.info("Set tokenizer padding_side to left for flash_attention_2")

    # --- PEFT/LoRA Handling (Assuming always enabled) ---
    # Check if model_name_or_path points to a downloaded adapter checkpoint
    if loading_existing_adapter:
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
            # Cast model to PreTrainedModel to satisfy type checker
            model = get_peft_model(model, peft_config)  # type: ignore # Cast might not fully resolve, ignore for now
            logger.info("Applied new PEFT config after adapter load failure.")
    else:
        # If no existing adapter checkpoint found, create and apply a new LoRA config
        # Only apply get_peft_model if not already loading an existing one
        if not loading_existing_adapter:
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
        else:
            logger.info(
                "Skipping new PEFT config application as an existing adapter was loaded."
            )

    # Print trainable parameters for verification
    model.print_trainable_parameters()  # type: ignore # Linter confusion

    logger.info("Model and processor loading complete.")

    return model, processor


def collate_fn(batch: List[Dict[str, Any]], processor: Any) -> Dict[str, torch.Tensor]:
    """Prepares a batch of data for the model."""
    processed_messages_list = []

    for item in batch:
        messages = item.get("messages", [])

        messages = oai_to_qwen(messages)
        processed_messages_list.append(messages)

    print("processed messages", processed_messages_list)
    # Filter out empty messages before proceeding
    valid_indices = [i for i, msg in enumerate(processed_messages_list) if msg]
    if not valid_indices:
        logger.warning("Collate function resulted in an empty batch.")
        # Return an empty dictionary or raise an error, depending on desired behavior
        return {}

    processed_messages_list = [processed_messages_list[i] for i in valid_indices]

    # Apply chat template and tokenize
    # Note: add_generation_prompt=False because we provide the full conversation including the assistant's response for training.
    tokenizer = getattr(processor, "tokenizer", processor)
    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in processed_messages_list
    ]
    print("texts", texts)

    # Process images - this assumes a processor like Qwen-VL's
    try:
        # Instead of directly using image_list
        image_inputs, _ = process_vision_info(processed_messages_list)

        # Attempt multimodal processing
        inputs = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            max_length=32768,
            truncation=True,
        )
    except Exception as e:
        logger.error(f"Failed to process batch: {e}")
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

    # Make absolutely sure we're correctly masking image tokens in labels
    # Directly set the hardcoded token IDs for Qwen2.5-VL
    image_token_ids = [151652, 151653, 151654, 151655, 151656]
    logger.info(f"Using hardcoded image token IDs for Qwen2.5-VL: {image_token_ids}")

    for img_tok_id in image_token_ids:
        # Ensure each image token is masked in labels
        mask = labels == img_tok_id
        if mask.any():
            labels[mask] = -100

    inputs["labels"] = labels

    # Debug info
    non_pad_tokens = (labels != -100).sum().item()
    logger.info(f"Batch has {non_pad_tokens} non-masked tokens for training")

    return inputs


def train(
    model: Union[PreTrainedModel, PeftModel, PeftMixedModel],
    processor: Any,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    output_dir: str,
    epochs: int,
    max_grad_norm: float,
    initial_loss: Optional[float] = None,
    best_eval_loss: float = float("inf"),
) -> Tuple[Dict[str, float], float, Optional[float]]:
    """
    Standard training loop (no gradient accumulation).

    Args:
        model (Union[PreTrainedModel, PeftModel, PeftMixedModel]):
            The model to train.
        processor (Any):
            The tokenizer/processor used to process inputs and possibly hold the tokenizer config.
        train_dataloader (DataLoader):
            The DataLoader providing the training batches.
        optimizer (torch.optim.Optimizer):
            An initialized optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR):
            An initialized scheduler.
        output_dir (str):
            Directory to save final and best model checkpoints.
        epochs (int):
            Number of training epochs.
        max_grad_norm (float):
            Maximum gradient norm to clip to.
        initial_loss (float, optional):
            If an initial loss is known (e.g., from a warmup run), pass it here.
        best_eval_loss (float, optional):
            If a best eval loss is known, pass it here.

    Returns:
        (Tuple[Dict[str, float], float, Optional[float]]):
            - metrics: A dictionary with final_loss, best_eval_loss (if found), and
              a possible loss_reduction_percent if initial_loss was known.
            - best_eval_loss: The updated best evaluation loss after training.
            - initial_loss: Possibly updated if it was None initially (and thus set).
    """
    logger.info("Starting training function...")

    num_training_steps_per_epoch = len(train_dataloader)
    total_training_steps = epochs * num_training_steps_per_epoch

    logger.info(f"Total training steps (all epochs): {total_training_steps}")
    logger.info(f"Current scheduler state: {scheduler.state_dict()}")

    # If using Qwen-VL or flash-attention, ensure left padding is set
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(model.config, "attn_implementation", None) == "flash_attention_2":
        tokenizer.padding_side = "left"
        logger.info("Set tokenizer padding_side to left for flash_attention_2")

    # Put model in training mode once at the start
    model.train()

    # Global trackers
    global_step = 0
    optimizer_step = (
        scheduler.state_dict().get("last_epoch", 0) * num_training_steps_per_epoch
    )
    logger.info(f"Attempting to resume optimizer step count at: {optimizer_step}")

    # We store rolling losses and best eval
    all_losses = []
    epoch_losses = []
    recent_losses = []

    logger.info("Starting training loop...")

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        num_batches_this_epoch = 0

        progress_bar = tqdm(
            total=num_training_steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{epochs}",
            position=0,
            leave=True,
        )

        for step, batch in enumerate(train_dataloader):
            if not batch:
                logger.warning("Empty batch received, skipping.")
                continue

            device = model.device
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Record raw loss
            current_loss = loss.item()
            all_losses.append(current_loss)

            if epoch == 0 and step == 0 and initial_loss is None:
                initial_loss = current_loss
                logger.info(f"Initial loss recorded: {initial_loss:.4f}")

            # Rolling average for display
            recent_losses.append(current_loss)
            if len(recent_losses) > 20:
                recent_losses.pop(0)
            loss_moving_avg = sum(recent_losses) / len(recent_losses)

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            optimizer_step += 1
            num_batches_this_epoch += 1
            epoch_loss += current_loss

            progress_bar.set_postfix(
                {
                    "loss": f"{loss_moving_avg:.4f}",
                    "last_loss": f"{current_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": optimizer_step,
                }
            )
            progress_bar.update()

            # Evaluate at ~every 10% of epoch steps
            if optimizer_step % max(1, num_training_steps_per_epoch // 10) == 0:
                logger.info(f"Running evaluation at step {optimizer_step}")

                # Temporarily set to eval mode
                was_training = model.training
                model.eval()

                with torch.no_grad():
                    outputs = model(**batch)
                    eval_loss = outputs.loss.item()
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")

                    if initial_loss is not None:
                        overall_change = (
                            (initial_loss - eval_loss) / initial_loss
                        ) * 100
                        logger.info(
                            f"Overall progress: {overall_change:.2f}% change from initial loss"
                        )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        logger.info(f"New best evaluation loss: {best_eval_loss:.4f}")

                        best_model_dir = os.path.join(output_dir, "best")
                        os.makedirs(best_model_dir, exist_ok=True)
                        model.save_pretrained(best_model_dir)
                        logger.info(f"Saved best model to {best_model_dir}")

                        # Also save optimizer/scheduler states
                        optimizer_save_path = os.path.join(
                            best_model_dir, "optimizer.pt"
                        )
                        scheduler_save_path = os.path.join(
                            best_model_dir, "scheduler.pt"
                        )
                        torch.save(optimizer.state_dict(), optimizer_save_path)
                        torch.save(scheduler.state_dict(), scheduler_save_path)
                        logger.info(
                            f"Saved final optimizer state to {optimizer_save_path}"
                        )
                        logger.info(
                            f"Saved final scheduler state to {scheduler_save_path}"
                        )

                if was_training:
                    model.train()

        progress_bar.close()

        # Compute average epoch loss
        if num_batches_this_epoch > 0:
            avg_epoch_loss = epoch_loss / num_batches_this_epoch
            epoch_losses.append(avg_epoch_loss)
            logger.info(
                f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}"
            )
            if len(epoch_losses) > 1:
                prev_epoch_loss = epoch_losses[-2]
                epoch_change = (
                    (prev_epoch_loss - avg_epoch_loss) / prev_epoch_loss
                ) * 100
                logger.info(
                    f"Epoch-to-epoch change: {epoch_change:.2f}% "
                    f"{'improvement' if epoch_change > 0 else 'worse'}"
                )
        else:
            logger.warning(f"Epoch {epoch + 1} had 0 batches processed!")

    # End of all epochs
    logger.info("Training completed!")

    if epoch_losses:
        final_epoch_loss = epoch_losses[-1]
        logger.info(f"Final epoch loss: {final_epoch_loss:.4f}")
        if len(epoch_losses) > 1:
            total_change = (
                (epoch_losses[0] - final_epoch_loss) / epoch_losses[0]
            ) * 100
            logger.info(f"Total training progress: {total_change:.2f}% improvement")
    else:
        final_epoch_loss = float("nan")

    # Save final model and processor
    logger.info(f"Saving final model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Model and processor saved.")

    metrics = {"final_loss": final_epoch_loss}
    if best_eval_loss < float("inf"):
        metrics["best_eval_loss"] = best_eval_loss
    if initial_loss is not None and not math.isnan(final_epoch_loss):
        metrics["loss_reduction_percent"] = (
            (initial_loss - final_epoch_loss) / initial_loss
        ) * 100

    return metrics, best_eval_loss, initial_loss


def evaluate_model(
    model: Union[PreTrainedModel, PeftModel, PeftMixedModel],
    processor: Any,
    batch: Optional[Dict[str, torch.Tensor]] = None,
) -> float:
    """Simple evaluation function to check model performance during training"""
    with torch.no_grad():
        if batch is not None:
            # Use the current batch for quick evaluation
            outputs = model(**batch)
            return outputs.loss.item()
        else:
            # In a real implementation, you would use a separate validation dataloader
            return 0.0


setup_script = """
pip install -q -U transformers datasets torch Pillow requests pydantic tqdm accelerate sentencepiece nebu peft bitsandbytes qwen-vl-utils
pip install flash-attn --no-build-isolation
"""


class TrainingRequest(BaseModel):
    model: str
    dataset: str
    adapter_name: str
    owner: Optional[str] = None
    learning_rate: float = 5e-5
    epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
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

    # --- Directory Cleanup ---
    local_model_dir = "/tmp/model_checkpoint"  # Local dir for loading/saving
    if os.path.exists(local_model_dir):
        logger.info(f"Removing existing local model directory: {local_model_dir}")
        try:
            shutil.rmtree(local_model_dir)
        except OSError as e:
            logger.error(f"Error removing directory {local_model_dir}: {e}")
            # Decide if this is fatal or if we can proceed cautiously
            # For now, we'll raise an error if cleanup fails.
            raise RuntimeError(f"Failed to clean up {local_model_dir}") from e
    os.makedirs(local_model_dir, exist_ok=True)  # Recreate after removal
    # --- End Directory Cleanup ---

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
    resuming_training = False  # Flag to indicate if we are resuming

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
                resuming_training = True  # Set flag
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
        existing_adapter_metadata = None  # Ensure it's None if no cache hit

    # --- Garbage Collection --- (Attempt before model loading)
    logger.info("Performing garbage collection before model loading...")
    gc.collect()
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()
    # --- End Garbage Collection ---

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

        # Print the first 5 examples
        logger.info("--- First 5 examples from the dataset ---")
        for i in range(min(5, len(train_dataset))):  # Ensure we don't go out of bounds
            try:
                example = train_dataset[i]
                logger.info(
                    f"Example {i+1}:\n{json.dumps(example, indent=2)}"
                )  # Pretty print JSON
            except Exception as e:
                logger.warning(f"Could not retrieve or print example {i+1}: {e}")
        logger.info("--- End of first 5 examples ---")

    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise RuntimeError("Failed to load dataset") from e

    logger.info("Creating dataloader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_request.batch_size,
        collate_fn=lambda batch: collate_fn(batch, processor),
        shuffle=True,
    )

    # --- Optimizer and Scheduler Setup ---
    logger.info("Setting up optimizer and scheduler...")
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        LambdaLR,  # Import for type hint consistency if needed
    )
    from transformers import get_cosine_schedule_with_warmup

    # Determine initial learning rate
    initial_lr = train_request.learning_rate
    if resuming_training and existing_adapter_metadata:
        initial_lr = existing_adapter_metadata.learning_rate
        logger.info(
            f"Resuming training. Using learning rate from existing adapter: {initial_lr}"
        )
    else:
        logger.info(
            f"Starting new training. Using learning rate from request: {initial_lr}"
        )

    # Check for trainable parameters BEFORE creating optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.error(
            "No trainable parameters found! PEFT/LoRA might not be configured correctly."
        )
        raise ValueError("No trainable parameters found in the model.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Setup scheduler - Calculate total steps based on CURRENT request's epochs
    # Note: If resuming, the scheduler's internal step count will be loaded later.
    num_training_steps_per_epoch = len(train_dataset)  # or len(train_dataloader)
    total_training_steps = train_request.epochs * num_training_steps_per_epoch
    num_warmup_steps = int(total_training_steps * 0.15)  # Use 15% warmup for this run

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,  # Scheduler adapts based on loaded state's step count
    )
    # Cast scheduler type explicitly for type checker if needed, though often inferred correctly
    scheduler: LambdaLR = scheduler  # type: ignore
    logger.info(
        f"Scheduler configured for {total_training_steps} total steps this run ({num_warmup_steps} warmup)."
    )

    # --- Load Optimizer and Scheduler State if Resuming ---
    if resuming_training:
        optimizer_load_path = os.path.join(local_model_dir, "optimizer.pt")
        scheduler_load_path = os.path.join(local_model_dir, "scheduler.pt")
        logger.info(f"Attempting to load optimizer state from: {optimizer_load_path}")
        logger.info(f"Attempting to load scheduler state from: {scheduler_load_path}")

        # Determine device for loading optimizer state
        # Use the device of the first model parameter if possible, otherwise default to CPU and let optimizer handle moving state
        try:
            map_location = next(model.parameters()).device
        except StopIteration:
            map_location = torch.device("cpu")
            logger.warning(
                "Could not determine model device, loading optimizer state to CPU."
            )

        if os.path.exists(optimizer_load_path):
            try:
                optimizer.load_state_dict(
                    torch.load(optimizer_load_path, map_location=map_location)
                )
                logger.info(
                    f"Successfully loaded optimizer state from {optimizer_load_path}"
                )
                # After loading state, ensure optimizer's parameters are on the correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(map_location)
                logger.info(
                    f"Moved loaded optimizer state tensors to device: {map_location}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load optimizer state from {optimizer_load_path}: {e}. Continuing with fresh optimizer state."
                )
        else:
            logger.warning(
                f"Optimizer state file not found at {optimizer_load_path}. Starting with fresh optimizer state."
            )

        if os.path.exists(scheduler_load_path):
            try:
                scheduler.load_state_dict(torch.load(scheduler_load_path))
                logger.info(
                    f"Successfully loaded scheduler state from {scheduler_load_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load scheduler state from {scheduler_load_path}: {e}. Continuing with fresh scheduler state."
                )
        else:
            logger.warning(
                f"Scheduler state file not found at {scheduler_load_path}. Starting with fresh scheduler state."
            )
    else:
        logger.info(
            "Not resuming training, using fresh optimizer and scheduler states."
        )

    # --- Training ---
    logger.info("Starting training process...")
    # Initialize best_eval_loss and initial_loss for the first run or load if possible
    best_eval_loss = float("inf")
    initial_loss = None  # Will be set in the first batch if not resuming effectively
    # We could try loading these from a saved metrics file too, but let's keep it simple for now.

    metrics, best_eval_loss, initial_loss = (
        train(  # Capture updated best_eval_loss and initial_loss
            model,
            processor,
            train_dataloader,
            optimizer,  # Pass instance
            scheduler,  # Pass instance
            output_dir=local_model_dir,
            # learning_rate=initial_lr, # Pass determined LR
            epochs=train_request.epochs,
            max_grad_norm=train_request.max_grad_norm,
            # gradient_accumulation_steps=train_request.gradient_accumulation_steps,
            initial_loss=initial_loss,  # Pass current initial_loss
            best_eval_loss=best_eval_loss,  # Pass current best_eval_loss
        )
    )
    final_loss = metrics.get("final_loss", float("nan"))
    logger.info(f"Training finished. Final loss: {final_loss:.4f}")

    # --- Single Example Evaluation ---
    logger.info("Running evaluation on a single training example...")
    if len(train_dataset) > 0:
        model.eval()  # Set model to evaluation mode
        try:
            eval_sample = train_dataset[0]
            logger.info(f"Raw evaluation sample:\n{json.dumps(eval_sample, indent=2)}")

            # Prepare input similar to inference
            eval_messages_oai = eval_sample.get("messages", [])
            if eval_messages_oai:
                # Convert the full conversation first for vision processing
                full_eval_messages_qwen = oai_to_qwen(eval_messages_oai)
                logger.info(
                    f"Full Qwen formatted messages for eval context:\n{full_eval_messages_qwen}"
                )

                try:
                    # Handle different versions of process_vision_info (might return 2 or 3 values)
                    vision_info = process_vision_info(full_eval_messages_qwen)
                    if len(vision_info) == 3:
                        image_inputs, video_inputs, _ = vision_info
                    else:
                        image_inputs, video_inputs = vision_info

                    if image_inputs:
                        logger.info(
                            f"Processing {len(image_inputs)} image(s) for eval."
                        )
                    if video_inputs:
                        logger.info(
                            f"Processing {len(video_inputs)} video(s) for eval."
                        )

                    # Prepare messages for the prompt (exclude assistant's turn)
                    prompt_messages_qwen = []
                    for msg in full_eval_messages_qwen:
                        prompt_messages_qwen.append(msg)
                        if msg.get("role") == "assistant":
                            prompt_messages_qwen.pop()  # Remove the assistant message we just added
                            break  # Stop after the turn *before* the first assistant message

                    if not any(
                        msg.get("role") == "user" for msg in prompt_messages_qwen
                    ):
                        logger.warning(
                            "No user messages found in prompt messages for eval, skipping."
                        )

                    else:
                        # Get the appropriate tokenizer
                        tokenizer = getattr(processor, "tokenizer", processor)
                        text = tokenizer.apply_chat_template(
                            prompt_messages_qwen,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        logger.info(
                            f"Input text for eval generation (excluding assistant answer):\n{text}"
                        )

                        try:
                            inputs = processor(
                                text=[text],  # Don't wrap in list to avoid type issues
                                images=image_inputs,  # Use vision info from full conversation
                                videos=video_inputs
                                if video_inputs
                                else None,  # Use vision info from full conversation
                                padding=True,
                                return_tensors="pt",
                            )

                            device = model.device
                            assert isinstance(
                                device, torch.device
                            ), f"Expected torch.device, got {type(device)}"
                            inputs = {
                                k: v.to(device)
                                for k, v in inputs.items()
                                if isinstance(v, torch.Tensor)
                            }

                            # Generate
                            with torch.no_grad():
                                # Limit max_new_tokens for quick eval
                                generated_ids = model.generate(
                                    input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    pixel_values=inputs.get("pixel_values"),
                                    image_grid_thw=inputs.get("image_grid_thw"),
                                    max_new_tokens=150,
                                )

                            # Decode
                            input_ids_length = inputs["input_ids"].shape[1]
                            generated_ids_trimmed = generated_ids[
                                :, input_ids_length:
                            ]  # Slice output
                            output_text = tokenizer.batch_decode(
                                generated_ids_trimmed, skip_special_tokens=True
                            )[0]

                            logger.info("--- Generated Output for Eval Sample ---")
                            logger.info(output_text)
                            logger.info("--- End Generated Output ---")
                        except Exception as e:
                            logger.error(f"Error during generation: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing vision info: {e}", exc_info=True)
            else:
                logger.warning(
                    "First training sample has no messages, skipping evaluation."
                )

        except Exception as e:
            logger.error(f"Error during single example evaluation: {e}", exc_info=True)
        finally:
            model.train()  # Set model back to training mode (good practice)
    else:
        logger.warning("Training dataset is empty, skipping single example evaluation.")
    # --- End Single Example Evaluation ---

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
        base_model=train_request.model
        if not existing_adapter_metadata
        else existing_adapter_metadata.base_model,  # Use original base model
        epochs_trained=total_epochs_trained,
        last_trained=int(time.time()),
        lora_rank=train_request.lora_rank,
        lora_alpha=train_request.lora_alpha,
        lora_dropout=train_request.lora_dropout,
        lora_target_modules=train_request.lora_target_modules,
        learning_rate=initial_lr,  # Save the learning rate used for this run
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
    # training_request = TrainingRequest(
    #     model="Qwen/Qwen2.5-VL-7B-Instruct",
    #     dataset="https://storage.googleapis.com/orign/testdata/nebu/clinton.jsonl",
    #     adapter_name="bar1",
    #     gradient_accumulation_steps=4,  # Example with accumulation
    # )

    training_request = TrainingRequest(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        dataset="https://storage.googleapis.com/orign/testdata/nebu/clinton.jsonl",
        adapter_name="bar20",
        gradient_accumulation_steps=8,
    )
    print("Training request: ", training_request.model_dump_json(indent=2))

    train_qwen_vl(training_request)
    print("Training job launched")
