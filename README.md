# FreeProtMap

## Prepare Datasets  
### Traing dataset
All raw traning datasets could be download from https://yanglab.nankai.edu.cn/trRosetta/benchmark/ .  
To generate their feature representation, run  
```
python /representation_generation/representation_generation.py  
```
Note: you have to set the path in representation_generation.py  
### Test datasets  
All test data could be downloaded from https://drive.google.com/drive/folders/15QIdVAUAITGc4zeakZpRBdyyUX2vJEzg?usp=share_link .  
They should be placed in this directory:  /datasets.



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
