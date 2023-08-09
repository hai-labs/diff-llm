.PHONY: twine tests clean clean-pyc upload-pypi-test upload-pypi

twine:
	pip install twine

clean:
	python setup.py clean

clean-pyc:
	find . -name '*.pyc' -exec rm {} \;

upload-pypi-test: twine
	python setup.py sdist bdist_wheel && \
		twine upload --repository-url https://test.pypi.org/legacy/ dist/* && \
		rm -rf dist

upload-pypi: twine
	python setup.py sdist bdist_wheel && \
		twine upload dist/* && \
		rm -rf dist

# Experiments
.PHONY: fine-tuning-med-0
fine-tuning-med-0:
	python diff_llm/fine_tune.py \
		--model-path EleutherAI/pythia-70m \
		--data-dir=datasets/diff_corpus_medium \
		--output-dir=models/diff_model_medium-0 \
		--report-to wandb \
		--max-length 512 \
		--gradient-accumulation-steps 2
