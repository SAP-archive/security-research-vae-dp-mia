folder := configs/attack_configs configs/model_configs configs/optimizer_configs experiments data models logs notebooks custom_models

targets := install folders

build_targets := build-migan

.PHONY: $(targets) clean $(build_targets)

all: $(targets)

buildall: $(build_targets)

install: requirements.txt folders
	python -m pip install -r requirements.txt

folders:
	mkdir -p $(folder)

clean:
	rm -r $(folder)

download-vgg16: | folders
	curl -o custom_models/rcmalli_vggface_tf_notop_vgg16.h5 -L https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5

download-lpips: | folders
	curl -o custom_models/net-lin_alex_v0.1_27.pb -L http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1_27.pb
	curl -o custom_models/net-lin_alex_v0.1.pb -L http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb

download-motionsense: | folders
	curl -o data/A_DeviceMotion_data.zip -L https://github.com/mmalekzadeh/motion-sense/blob/master/data/A_DeviceMotion_data.zip?raw=true
	unzip -d data data/A_DeviceMotion_data.zip
	rm data/A_DeviceMotion_data.zip
	rm -r data/__MACOSX

	curl -o data/A_DeviceMotion_data/data_subjects_info.csv -L https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv
