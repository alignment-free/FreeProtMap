# FreeProtMap

## Data Preparation  
All raw traning datasets could be download from https://yanglab.nankai.edu.cn/trRosetta/benchmark/ .  
To obtain their feature representation, run  
```
python /representation_generation/representation_generation.py  
```
Note: you have to set the path in representation_generation.py  
  
All test data could be downloaded from https://drive.google.com/file/d/15eF6yrRyJx0KvnOMTpl_i7TVrrk-KuxQ/view?usp=sharing .  
They should be placed under the /datasets.



## Evaluation
To evaluate, you need to enter the corresponding directory first  
```
cd /test
```

To evaluate the performance of distance prediction, run
```
python Eval_distance.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path
```

To evaluate the performance of contact prediction, run
```
python Eval_contact_value.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path  
python Eval_contact_curve.py  -weight your_weight_path  -feature your your_feature_path  -label  your_label_path
```
