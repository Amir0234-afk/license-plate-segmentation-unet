.PHONY: setup train predict clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.train_unet --images data/images --masks data/masks --outdir models --epochs 5

predict:
	python -m src.predict_unet --images data/test --model models/unet.keras --out results --overlay

clean:
	rm -rf models results __pycache__ .venv
