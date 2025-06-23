from sub.test_processor import (
    Input,
    my_async_generator_processor,
    my_async_processor,
    my_generator_processor,
)

out = my_async_processor(Input(greeting="Hello, world!"), poll=True, wait=True)

print(out)

out = my_async_processor(Input(greeting="Yellow world?!"), poll=True, wait=True)

print(out)

# --- Streaming generator call ---
print("\n--- Streaming generator test ---\n")

for chunk in my_generator_processor(Input(greeting="Stream me!"), stream=True):
    print("received chunk:", chunk)

print("\n--- Streaming async generator test ---\n")
for chunk in my_async_generator_processor(Input(greeting="Stream me!"), stream=True):
    print("received chunk:", chunk)

# --- Async streaming generator call ---
print("\n--- Streaming async generator test ---\n")

for chunk in my_async_generator_processor(
    Input(greeting="Async stream me!"), stream=True
):
    print("received chunk:", chunk)
