from chatmux.openai import ChatRequest
from infer import infer_qwen_vl

req = ChatRequest(
    model="clinton16",
    messages=[{"role": "user", "content": "Who is this an image of?"}],
)

infer_qwen_vl(req)
