.PHONY: test clean docs

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1
# install only Lightning Trainer packages
export PACKAGE_NAME=pytorch

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf $(shell find . -name "lightning_logs")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source-fabric/api/generated
	rm -rf ./docs/source-pytorch/notebooks
	rm -rf ./docs/source-pytorch/generated
	rm -rf ./docs/source-pytorch/*/generated
	rm -rf ./docs/source-pytorch/api
	rm -rf ./docs/source-app/generated
	rm -rf ./docs/source-app/*/generated
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

docs:
	git submodule update --init --recursive # get Notebook submodule
	pip install -qq lightning # install (stable) Lightning from PyPI instead of src
	pip install -qq -r requirements/app/base.txt
	pip install -qq -r requirements/pytorch/docs.txt
	cd docs/source-pytorch && $(MAKE) html --jobs $(nproc) && cd ../../

update:
	git submodule update --init --recursive --remote
