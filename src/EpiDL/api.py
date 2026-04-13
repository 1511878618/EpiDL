from simulation_infective_diseases.Model import fit_sir_etc_model, fit_dl_ml_model
import pandas as pd 
SIR_ETC_Model = ['SIR','SEIR']
DL_ML_Model = ['LSTM', 'GRU', 'CNN', 'Dlinear']
def fit_model(data, model_kwargs, Time_col, Case_col, *args, **kwargs):
    """
    
    data_kwargs = {
        "Time_col":?
        "Case_col":?
    }
    """
    # check modelType
    modelType = model_kwargs.get("modelType", None)


    data = data[[Time_col, Case_col]].copy().dropna()
    ## preprocess value type
    data[Time_col] = pd.to_datetime(data[Time_col], errors='coerce') # sort
    data[Case_col] = pd.to_numeric(data[Case_col], errors='coerce') # set to values
    data = data.sort_values(Time_col)
    # fit model 
    if modelType in SIR_ETC_Model:
        fit_res = fit_sir_etc_model(data, Time_col = Time_col, Case_col = Case_col, model_kwargs = model_kwargs)
    elif modelType in DL_ML_Model:
        fit_res = fit_dl_ml_model(data, Time_col = Time_col, Case_col = Case_col, model_kwargs = model_kwargs)
    else:
        raise KeyError(f"Model type is not supported for {modelType}")
    
    # return results 
    return {"Model":fit_res.get("Model", None), "Prediction_df":fit_res.get("Prediction_df", None), "Metric": fit_res.get("Metric", None)   }
    
