from Eir import SEIR, SIR
import Eir
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd 
def fit_sir_etc_model(data, Time_col, Case_col,model_kwargs, **kwargs):
    """
    fit SIR model, supported others from package of Eir
    """
    # do preprocess 
    modelType = model_kwargs.pop("modelType")
    data = data[[Time_col, Case_col]].copy().dropna()
    
    # define timesteps and dt
    timesteps = data.shape[0]-1  # 
    dt = .1 # 提高颗粒度

    # model_kwargs = {k: float(v) if isinstance(v, np.float64) else v for k, v in model_kwargs.items()}

    sim = Eir.__dict__[modelType](**model_kwargs)

    df = sim.run(timesteps, dt, plot=False)
    # keep dt of each timesteps respondce to timesteps
    df = df.iloc[::int(1/dt)]

    df['Pred'] = df['Infected']

    data = pd.concat([data.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    
    # evaluate
    Metric = {} 
    Metric['MSE'] = mean_squared_error(
        y_true = data[Case_col],
        y_pred= data['Pred']
    )
    Metric['MAE'] = mean_absolute_error(
        y_true = data[Case_col],
        y_pred= data['Pred']
    )
    return {
        "Model":sim, 
        "Prediction_df":data, 
        "Metric":Metric
    }
     
