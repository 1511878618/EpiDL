
## Installation

### 1. Clone the repository

```bash
git clone git@github.com:1511878618/EpiDL.git
cd EpiDL
```

### 2. Create a conda environment

```bash
conda create -n epidl python=3.12
conda activate epidl
```

### 3. Install dependencies

```bash
pip install epilearn
pip install Eir
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
pip install torch_geometric
pip install epydemix
```

### 4. Install EpiDL in editable mode

```bash
pip install -e .
```


## Example

```
# 输入示例数据
#	Year/Month	Cases/pct
# 0	2014-07-14	1.0
# 1	2014-07-21	NaN
# 2	2014-07-28	0.0
# 3	2014-08-04	1.0
# 4	2014-08-11	0.0

import EpiDL

# 需要的参数
time_col = 'Year/Month'
Case_col = 'Cases/pct'

CNN_default_params = {
    "modelType":"CNN",
    "num_timesteps_input":12,
    "num_timesteps_output":3,
    "lr":1e-3,
    "train_rate":0.6,
    "val_rate":0.2,
    "weight_decay":1e-6,
    "epochs":100,
    "batch_size":32,
    "linear_hid":256,
    "dropout":.5
}
# 运行模型拟合
fit_res = EpiDL.api.fit_model(
    data = data, 
    Time_col = time_col,
    Case_col = Case_col,
    model_kwargs =params
)
# fit_res是字典，包含'Model', 'Prediction_df', 'Metric'三个key
# Model 是模型对象，需要保存
# Prediction_df 是预测结果，需要保存
# Metric 是模型指标，需要保存
print(fit_res)

# dict_keys(['Model', 'Prediction_df', 'Metric'])

print(fit_res['Prediction_df'])

# Year/Month  Cases/pct  Type  Pred
# 591 2025-11-10       20.0  Test    10.452312
# 592 2025-11-17       30.0  Test    33.484791
# 593 2025-11-24       31.0  Test    29.570511
# 594 2025-12-01       37.0  Test    26.192924
# 595 2025-12-08        4.0  Test    17.283758
# 596 2025-12-15        6.0  Test    10.040459
# 597 2025-12-22        5.0  Test     5.432630
# 598 2025-12-29        3.0   NaN          NaN
# 599 2026-01-05        0.0   NaN          NaN
# 600 2026-01-12        2.0   NaN          NaN
```