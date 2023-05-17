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
ESM,dlogger,openfold need to be installed.
'''
git clone https://github.com/facebookresearch/esm.git  
git clone https://github.com/NVIDIA/dllogger.git
git clone https://github.com/aqlaboratory/openfold.git
'''
Enter the corresponding file directory and run 
'''
pip install .
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
