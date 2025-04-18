from infer import infer_qwen_vl

from nebu.chatx.openai import ChatCompletionRequest

req = ChatCompletionRequest(
    model="clinton16",
    messages=[{"role": "user", "content": "Who is this an image of?"}],
)

infer_qwen_vl(req)
