from train import TrainingRequest, train_unsloth_sft

training_req = TrainingRequest(
    adapter_name="clinton16",
    dataset="https://storage.googleapis.com/orign/testdata/nebu/clinton.jsonl",
)

train_unsloth_sft(training_req)
