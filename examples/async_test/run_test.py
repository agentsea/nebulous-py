from examples.async_test.sub.test_processor import Input, my_async_processor

out = my_async_processor(Input(greeting="Hello, world!"), poll=True, wait=True)

print(out)

out = my_async_processor(Input(greeting="Yellow world?!"), poll=True, wait=True)

print(out)
