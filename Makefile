all: build install

build:
	bash release/docker/obfuscate_source_code.sh
	python3 setup.py bdist_wheel
	bash release/docker/revert_obfuscation.sh

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

install: build
	pip3 install dist/nvidia_tao_tf2-*.whl

uninstall:
	pip3 uninstall -y nvidia-tao-tf2

