.PHONY: test
test:
	uv run pytest tests/ -v -s

.PHONY: generate-schema
generate-schema:
	rm -rf src/nebulous/chatx/openai.py
	uv run datamodel-codegen --input ./spec/openai.yaml --input-file-type openapi --output ./src/nebulous/chatx/openai.py --output-model-type pydantic_v2.BaseModel --snake-case-field --use-union-operator --reuse-model --target-python-version 3.11 --use-double-quotes --field-constraints 

.PHONY: run-async-test
run-async-test:
	uv run python -m examples.async_test.sub.test_processor