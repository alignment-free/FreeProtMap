# FreeProtMap

## Prepare Datasets  
### Traing dataset
Raw traning dataset could be download from https://yanglab.nankai.edu.cn/trRosetta/benchmark/ .  
To generate feature representation, run  
```
cd /representation_generation
python representation_generation.py  -source_dir your fatsa_path -save_dir  your save_path
```
### Test datasets  
All test data could be downloaded from https://drive.google.com/drive/folders/1oOVKjiTFtnetrInZyASEp7Y7yyF85lsv?usp=sharing .  
Save them under the path of datasets/ .


### Installation
Openfold needs to be installed first.
'''
absl-py==1.4.0
aliyun-python-sdk-core==2.13.36
aliyun-python-sdk-kms==2.16.0
antlr4-python3-runtime==4.9.3
bio==1.5.3
biopython==1.81
biothings-client==0.2.6
biotite==0.36.1
certifi==2022.12.7
cffi==1.15.1
charset-normalizer==3.0.1
contextlib2==21.6.0
contourpy==1.0.7
crcmod==1.7
cryptography==39.0.1
cycler==0.11.0
DLLogger @ file:///data/home/huangjiajian/chenmi/dllogger
dm-tree==0.1.8
einops==0.6.0
fair-esm @ file:///data/home/huangjiajian/chenmi/esm
fonttools==4.39.3
idna==3.4
importlib-resources==5.12.0
jmespath==0.10.0
joblib==1.2.0
kiwisolver==1.4.4
matplotlib==3.7.1
ml-collections==0.1.1
msgpack==1.0.4
mygene==3.2.2
networkx==3.0
numpy @ file:///home/conda/feedstock_root/build_artifacts/numpy_1675642561072/work
omegaconf==2.3.0
openfold @ file:///data/home/huangjiajian/chenmi/openfold
oss2==2.16.0
packaging==23.1
pandas==2.0.1
Pillow @ file:///home/conda/feedstock_root/build_artifacts/pillow_1675487166627/work
protobuf==3.20.3
pycparser==2.21
pycryptodome==3.17
pymesh2==0.3
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
PyYAML==6.0
requests==2.28.2
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
six==1.15.0
sklearn==0.0.post4
tensorboardX==2.6
threadpoolctl==3.1.0
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.0
tqdm==4.64.1
typing-extensions==3.7.4
tzdata==2023.3
urllib3==1.26.14
zipp==3.15.0
'''


## Evaluation
To evaluate FreeProtMap, run this command firstly
```
cd /test
```

To evaluate FreeProtMap on distance prediction task, run
```
python Eval_distance.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path
```

To evaluate FreeProtMap on contact prediction task, run
```
python Eval_contact_value.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path  
python Eval_contact_curve.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path
```
