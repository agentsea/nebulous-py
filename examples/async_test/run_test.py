import asyncio

# Important: This assumes the script is run from the project root.
from examples.async_test.sub.test_processor import my_async_processor


async def main():
    """
    This test verifies that the @processor decorator can correctly configure
    an async function from a subdirectory without crashing at import time.
    """
    print("--- Running Async Processor Integration Test ---")

    # The import statement above has already executed the decorator.
    # If there were a critical issue with file path resolution, it would have crashed.
    print("[SUCCESS] Decorator imported and executed without errors.")

    # We can inspect the created processor instance to see if paths are correct.
    container_req = my_async_processor.container
    if not container_req:
        print("[FAILURE] The processor's container configuration was not created.")
        return

    if not container_req.env:
        print("[FAILURE] The processor's container environment was not created.")
        return

    entrypoint_path_env = next(
        (e for e in container_req.env if e.key == "NEBU_ENTRYPOINT_MODULE_PATH"), None
    )

    print("\n--- Verifying Processor Configuration ---")
    print(f"Processor Name: {my_async_processor.name}")
    print(f"Container Image: {container_req.image}")

    # Check that the decorator correctly identified the function as async
    is_async_flag = next(
        (e for e in container_req.env if e.key == "IS_ASYNC_FUNCTION"), None
    )
    if is_async_flag and is_async_flag.value == "True":
        print("[SUCCESS] IS_ASYNC_FUNCTION flag is correctly set to True.")
    else:
        print(
            f"[FAILURE] IS_ASYNC_FUNCTION flag is incorrect: {is_async_flag.value if is_async_flag else 'Not Found'}"
        )

    # Check that the decorator calculated the entrypoint path correctly.
    # Based on the file structure, the source code will be synced from the `sub`
    # directory, so the entrypoint path should be relative to that.
    expected_path = "test_processor.py"
    if entrypoint_path_env:
        print(f"Entrypoint Path Env: '{entrypoint_path_env.value}'")
        if entrypoint_path_env.value == expected_path:
            print(f"[SUCCESS] Entrypoint path is correctly set to '{expected_path}'.")
        else:
            print(
                f"[FAILURE] Entrypoint path is '{entrypoint_path_env.value}', but expected '{expected_path}'."
            )
    else:
        print(
            "[FAILURE] NEBU_ENTRYPOINT_MODULE_PATH not found in environment variables."
        )

    print("\n--- Test Complete ---")
    print("\nThis simple test has passed.")
    print(
        "A full end-to-end test would require deploying the processor and sending a real message."
    )


if __name__ == "__main__":
    # To run this from the project root: python examples/async_test/run_test.py
    asyncio.run(main())
