# Variables
PYTHON = python3
GRPC_PORT = 50051

# Install dependencies
install:
	pip3 install -r requirements.txt

# Generate gRPC files
grpc:
	rm -f grpc_service/*_pb2.py grpc_service/*_pb2_grpc.py
	$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_service/*.proto

# Run gRPC Server
run-server:
	$(PYTHON) server.py

# Run gRPC Client
run-client:
	$(PYTHON) client.py

# Run tests
test:
	pytest tests/

lint:
	ruff check . --exclude venv --fix

format:
	ruff format . --exclude venv

# Run all checks
check: format lint test

# Clean up
clean:
	rm -rf __pycache__ grpc/*.pyc grpc/*_pb2.py grpc/*_pb2_grpc.py

# Default target
.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  grpc         - Generate gRPC code"
	@echo "  run-server   - Run gRPC server"
	@echo "  run-client   - Run gRPC client"
	@echo "  test         - Run tests"
	@echo "  format       - Format code using black"
	@echo "  lint         - Lint code using flake8"
	@echo "  check        - Run format, lint, and test"
	@echo "  clean        - Remove compiled Python files"
