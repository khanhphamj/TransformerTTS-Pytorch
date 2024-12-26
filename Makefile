.PHONY: install train

install:
	poetry install

train:
	poetry run python tts_model/train.py
