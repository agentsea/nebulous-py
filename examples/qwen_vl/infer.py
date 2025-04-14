from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nebu import Adapter, Message, processor
from nebu.chatx.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
)

setup_script = """
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/huggingface/transformers accelerate
pip install tiktoken peft bitsandbytes
pip install qwen-vl-utils[decord]==0.0.8
"""


def init():
    import time
    from dataclasses import dataclass
    from typing import Dict, Optional

    @dataclass
    class InferenceState:
        base_model_raw: Qwen2_5_VLForConditionalGeneration
        model_processor: AutoProcessor
        peft_model: Optional[PeftModel]
        base_model_id: str
        adapters: Dict[str, Adapter]

    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    print("loading model...")
    time_start_load = time.time()
    base_model_raw = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model_processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    print(f"Loaded model in {time.time() - time_start_load} seconds")

    global state
    state = InferenceState(
        base_model_raw=base_model_raw,
        model_processor=model_processor,
        peft_model=None,
        base_model_id=base_model_id,
        adapters={},
    )


@processor(
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
    setup_script=setup_script,
    accelerators=["1:A100_SXM"],
    platform="runpod",
    init_func=init,
)
def infer_qwen_vl(
    message: Message[ChatCompletionRequest],
) -> ChatCompletionResponse:
    import time

    full_time = time.time()
    import uuid

    from qwen_vl_utils import process_vision_info

    from nebu import (
        Adapter,
        Bucket,
        is_allowed,
        oai_to_qwen,
    )

    global state

    print("message", message)
    training_request = message.content
    if not training_request:
        raise ValueError("No training request provided")

    print("content", message.content)

    content = message.content
    if not content:
        raise ValueError("No content provided")

    adapter_hot_start = time.time()
    adapter_name_to_load = content.model

    if adapter_name_to_load not in state.adapters:
        print(
            f"Adapter '{adapter_name_to_load}' not loaded locally. Attempting to load."
        )

        mock_adapter_uri = f"s3://your-bucket/path/to/adapters/{adapter_name_to_load}"
        mock_base_model = "Qwen/Qwen-VL-Chat"
        mock_owner = "user_or_org_id"
        print(
            f"WARNING: Using mock adapter data for '{adapter_name_to_load}'. Replace with actual retrieval."
        )
        val = Adapter(
            name=adapter_name_to_load,
            uri=mock_adapter_uri,
            base_model=mock_base_model,
            owner=mock_owner,
            created_at=int(time.time()),
        )

        if not is_allowed(val.owner, message.user_id, message.orgs):
            raise ValueError(
                f"User not allowed to use adapter '{adapter_name_to_load}'"
            )

        if not val.base_model == state.base_model_id:
            raise ValueError(
                f"Adapter base model '{val.base_model}' does not match loaded base model '{state.base_model_id}'"
            )

        adapter_path = f"./adapters/{adapter_name_to_load}"
        bucket = Bucket()
        print("copying adapter", val.uri, adapter_path)
        time_start = time.time()
        try:
            bucket.copy(val.uri, adapter_path)
            print(f"Copied adapter files in {time.time() - time_start} seconds")
        except Exception as e:
            raise RuntimeError(
                f"Failed to copy adapter from {val.uri} to {adapter_path}: {e}"
            )

        print("loading adapter into model", adapter_name_to_load)
        load_adapter_start = time.time()
        if state.peft_model is None:
            print("Loading first adapter, creating PeftModel...")
            state.peft_model = PeftModel.from_pretrained(
                state.base_model_raw,
                adapter_path,
                adapter_name=adapter_name_to_load,
                is_trainable=False,
            )
        else:
            print(f"Loading additional adapter '{adapter_name_to_load}'...")
            state.peft_model.load_adapter(
                adapter_path,
                adapter_name=adapter_name_to_load,
                is_trainable=False,
            )
        state.adapters[adapter_name_to_load] = val
        print(
            f"Loaded adapter '{adapter_name_to_load}' in {time.time() - load_adapter_start} seconds"
        )

    else:
        print(f"Adapter '{adapter_name_to_load}' already loaded.")
    print(f"Adapter check/load took {time.time() - adapter_hot_start} seconds")

    if state.peft_model is None:
        raise RuntimeError("PEFT model not initialized. No adapter loaded.")

    loaded_adapter_names = list(state.peft_model.peft_config.keys())
    print("Currently loaded adapter configurations: ", loaded_adapter_names)

    if adapter_name_to_load in loaded_adapter_names:
        print("setting active adapter:", adapter_name_to_load)
        state.peft_model.set_adapter(adapter_name_to_load)
    else:
        raise ValueError(
            f"Adapter '{adapter_name_to_load}' was requested but is not loaded in the PeftModel."
        )

    conent_dict = content.model_dump()
    messages_oai = conent_dict["messages"]
    messages = oai_to_qwen(messages_oai)

    print("preparing inputs using messages: ", messages)
    inputs_start = time.time()
    text = state.model_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("text: ", text)
    print("processing vision info: ", messages)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = state.model_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(state.peft_model.device)
    print("inputs device:", inputs.input_ids.device)
    print(f"Inputs prepared in {time.time() - inputs_start} seconds")

    generation_start = time.time()
    generated_ids = state.peft_model.generate(
        **inputs, max_new_tokens=content.max_tokens or 1024
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = state.model_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("output_text", output_text)
    print(f"Generation took {time.time() - generation_start} seconds")

    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=content.model,
        object="chat.completion",
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionResponseMessage(
                    role="assistant",
                    content=output_text[0],
                    refusal=None,
                    annotations=None,
                    function_call=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        service_tier=None,
        system_fingerprint=None,
        usage=None,
    )
    print(f"Total time: {time.time() - full_time} seconds")

    return response
