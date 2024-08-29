# CT项目检测网络记录

## 1. Command

1. luna16
- 单卡训练 luna16重采样数据集
```shell
CUDA_VISIBLE_DEVICES=0 python3 luna16_training.py -e ./config/environment_luna16_fold0.json -c ./config/config_train_luna16_16g.json
```

2. ct luna16 config 对角坐标
- 单卡训练 项目数据集
```shell
CUDA_VISIBLE_DEVICES=0 python3 luna16_training.py -e ./config/environment_ct.json -c ./config/config_train_ct.json
```

3. ct luna16 config_train_ct_improved 中心坐标
- 单卡训练 项目数据集
```shell
CUDA_VISIBLE_DEVICES=0 python3 luna16_training.py -e ./config/environment_ct.json -c ./config/config_train_ct_improved.json
```

```shell
CUDA_VISIBLE_DEVICES=1 python3 luna16_training.py -e ./config/environment_ct.json -c ./config/config_train_ct_improved.json
```

加入合成数据集训练 共计12000
```shell
CUDA_VISIBLE_DEVICES=2 python3 luna16_training.py -e ./config/environment_ct_concat.json -c ./config/config_train_ct_improved.json
```

4. test 测试指标

- 默认测试 输出json
```shell
CUDA_VISIBLE_DEVICES=0 python3 luna16_testing.py -e ./config/environment_ct.json -c ./config/config_train_ct_improved.json
```

- 指标测试
```shell
CUDA_VISIBLE_DEVICES=3 python3 luna16_testing_metric.py -e ./config/environment_ct_evaluate.json -c ./config/config_train_ct_improved.json
```

## 2. Memo

### 2024-08-29
- 等待显卡训练 合成数据

### 2024-08-27
- 加入合成数据集7000个（数据分布不均衡）真实数据集5000 训练 在第一批数据集测试

### 2024-08-25
- /opt/data/private/lwz/workplace/tutorials/detection/config/environment_ct.json 修改了model path和tensorboard path
- 修改了测试的class wise
- retinanet anchor base shape 保持不变
- nms thres 保持不变
- 尚未测试ct测试集 $\to$ 第二批


## 3. Results

### model_ct_2nd.pt 在第一批测试集的指标 Sun Aug 25 11:57:53 MSK 2024

> mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.10760738609804799
> gun_part_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.0054966999979626195
> gun_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.08649135182629061
> knife_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.008755648804636933
> lighter_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.0009183404307854203
> battery_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.43637488943056435
> 
> AP_IoU_0.10_MaxDet_100 -------- **0.14530168400321267**
> gun_part_AP_IoU_0.10_MaxDet_100 -------- 0.022203319438613287 **偏低**
> gun_AP_IoU_0.10_MaxDet_100 -------- 0.14472129822957633 **偏低**
> knife_AP_IoU_0.10_MaxDet_100 -------- 0.026544456075102387 **偏低**
> lighter_AP_IoU_0.10_MaxDet_100 -------- 0.004081192371057402 **偏低**
> battery_AP_IoU_0.10_MaxDet_100 -------- 0.5289581539017139
> 
> mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.33619983022411665
> gun_part_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.14043993378678957
> gun_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.13046595060990918
> knife_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.4053081406487359
> lighter_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.2834528111335304
> battery_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.7213323149416182
> 
> AR_IoU_0.10_MaxDet_100 -------- **0.587312975525856**
> gun_part_AR_IoU_0.10_MaxDet_100 -------- 0.3604061007499695 **偏低**
> gun_AR_IoU_0.10_MaxDet_100 -------- 0.20645160973072052 **偏低**
> knife_AR_IoU_0.10_MaxDet_100 -------- 0.8178137540817261
> lighter_AR_IoU_0.10_MaxDet_100 -------- 0.725806474685669
> battery_AR_IoU_0.10_MaxDet_100 -------- 0.8260869383811951

### model_ct.pt 在第一批测试集的指标 Sun Aug 25 12:14:12 MSK 2024

> mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.35090558187474113
> gun_part_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.19180527011422302
> gun_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.2589577359892223
> knife_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.30034824298250073
> lighter_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.1575745953526399
> battery_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.8458420649351197
> 
> AP_IoU_0.10_MaxDet_100 -------- **0.4630058792631815**
> gun_part_AP_IoU_0.10_MaxDet_100 -------- 0.278585439898295 **偏低**
> gun_AP_IoU_0.10_MaxDet_100 -------- 0.5721194617228933 **偏低**
> knife_AP_IoU_0.10_MaxDet_100 -------- 0.38294053398589095 **偏低**
> lighter_AP_IoU_0.10_MaxDet_100 -------- 0.2105670598252575 **偏低**
> battery_AP_IoU_0.10_MaxDet_100 -------- 0.8708169008835708
> 
> mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.6658193276988136
> gun_part_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.5408911440107558
> gun_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.41792114906840855
> knife_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.6905083159605662
> lighter_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.7120668987433115
> battery_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.9677091307110257
> 
> AR_IoU_0.10_MaxDet_100 -------- **0.8479103922843934**
> gun_part_AR_IoU_0.10_MaxDet_100 -------- 0.6802030205726624 **偏低**
> gun_AR_IoU_0.10_MaxDet_100 -------- 0.6935483813285828 **偏低**
> knife_AR_IoU_0.10_MaxDet_100 -------- 0.9210526347160339
> lighter_AR_IoU_0.10_MaxDet_100 -------- 0.9516128897666931
> battery_AR_IoU_0.10_MaxDet_100 -------- 0.9931350350379944

### model_ct.pt 在第一批和第二批测试集的指标 Sun Aug 25 13:33:13 MSK 2024

> mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.27291295015654243
> gun_part_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.1119842967643465
> gun_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.24267341637506473
> knife_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.23890680762169714
> lighter_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.1484416804177281
> battery_mAP_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.6225585496038756
> 
> AP_IoU_0.10_MaxDet_100 -------- **0.3894809745667078**
> gun_part_AP_IoU_0.10_MaxDet_100 -------- 0.173977529058362 **偏低**
> gun_AP_IoU_0.10_MaxDet_100 -------- 0.5952721149614542 **偏低**
> knife_AP_IoU_0.10_MaxDet_100 -------- 0.32197093052586706 **偏低**
> lighter_AP_IoU_0.10_MaxDet_100 -------- 0.19014155701042548 **偏低**
> battery_AP_IoU_0.10_MaxDet_100 -------- 0.6660427412774304
> 
> mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.5805363108714422
> gun_part_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.36516942580540973
> gun_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.42197253472275204
> knife_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.6647404763433669
> lighter_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.6624233888255225
> battery_mAR_IoU_0.10_0.50_0.05_MaxDet_100 -------- 0.7883757286601596
> 
> AR_IoU_0.10_MaxDet_100 -------- **0.7744613409042358**
> gun_part_AR_IoU_0.10_MaxDet_100 -------- 0.5234042406082153 **偏低**
> gun_AR_IoU_0.10_MaxDet_100 -------- 0.7253433465957642
> knife_AR_IoU_0.10_MaxDet_100 -------- 0.8905109763145447
> lighter_AR_IoU_0.10_MaxDet_100 -------- 0.8896746635437012
> battery_AR_IoU_0.10_MaxDet_100 -------- 0.8433734774589539
