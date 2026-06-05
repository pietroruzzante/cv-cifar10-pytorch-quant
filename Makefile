.PHONY: train evaluate quantize prune export all

RUN_ID ?= ""

train:
	cd src && python3 train.py

evaluate:
	cd src && python3 evaluate.py --run_id $(RUN_ID)

quantize:
	cd src && python3 quantize.py --run_id $(RUN_ID)

prune:
	cd src && python3 prune.py --run_id $(RUN_ID)

export:
	cd src && python3 export_onnx.py

all: train
	@echo "Training done. Set RUN_ID and run: make evaluate quantize prune"

mlflow:
	mlflow ui --backend-store-uri sqlite:///$(shell pwd)/src/mlflow.db --port 5001
