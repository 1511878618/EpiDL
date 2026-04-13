import torch 
from epilearn.data import UniversalDataset
from epilearn.utils import transforms
from epilearn.tasks import Forecast
# from epilearn.models.Temporal import LSTMModel, CNNModel, DlinearModel, VARMAXModel, ARIMAModel
import epilearn.models.Temporal as epilearn_Temporal
import pandas as pd 
SPECIAL_MODEL_PARAM_DICT = {
    "CNNModel":['linear_hid', 'dropout']
}


def find_exact_index_ranges(df, col, tgt_values):
    values = df[col].tolist()
    n = len(tgt_values)
    
    matches = []
    for i in range(len(values) - n + 1):
        if values[i:i+n] == tgt_values:
            # only can be matched once!
            return df.index[i], df.index[i+n-1]
    return None 

def fit_dl_ml_model(data, Time_col, Case_col, model_kwargs,device = 'cpu', **kwargs):
    """
    fit dl and ml model, currently not supported for optuna and other hyperparameters tuning framework.
    """
    # basie preprocess 
    modelType = model_kwargs.pop("modelType")
    data = data[[Time_col, Case_col]].copy().dropna() # This model do not supported for NA data

    # preprocess for epilearn 
    dataset = UniversalDataset(
    x=torch.Tensor(data[Case_col].values).unsqueeze(-1), 
                           y=torch.Tensor(data[Case_col].values).unsqueeze(-1), # prediction target
                           )
    
    # transforms
    # transformation = transforms.Compose({
    #             "features": [transforms.normalize_feat()], # Normalize the features
    #             "target":  [transforms.normalize_target()], # Normalize targets
    #                                 })
    # dataset.transforms = transformation

    # define model 
    prototype = modelType + "Model"
    pop_to_model_args = [
        'num_timesteps_input','num_timesteps_output',
    ] 
    # to adding into pop_to_model_args
    specical_model_args = SPECIAL_MODEL_PARAM_DICT.get(prototype, ['nhid','dropout'])
    pop_to_model_args +=specical_model_args 

    # get model_args
    model_args = {"device":device, 'num_features':1 # currently only based on infective counts of previous. 
                  }
    
   

    for model_args_key in pop_to_model_args:
        model_args[model_args_key] = model_kwargs.pop(model_args_key) # TODO: check key exists

    
    print(model_kwargs)
    # TODO: check all values 

    task = Forecast(prototype=epilearn_Temporal.__dict__[prototype],
                lookback=model_args['num_timesteps_input'],
                horizon=model_args['num_timesteps_output'],
                device=device)
    result = task.train_model(dataset=dataset,
                            loss='mse',      # loss function; using MSE as default
                            

                            model_args=model_args,
                            device='cpu',
                            **model_kwargs
                            )    # Using CPU could be slow though

    predictions, groundtruth = task.plot_forecasts(task.test_dataset, index_range=(0, -1))

    # GET IDX
    groundtruth = groundtruth.numpy()
    groundtruth = groundtruth.astype(data[Case_col].dtype)
    groundtruth = groundtruth.tolist()

    test_start_idx, test_end_idx =  find_exact_index_ranges(data, Case_col, tgt_values=groundtruth)

    data.loc[test_start_idx:test_end_idx, 'Type'] = 'Test'
    data.loc[test_start_idx:test_end_idx, 'Predictions'] = predictions.numpy()
    print(f"Test data range: {test_start_idx} to {test_end_idx}")

    Metric = result
    del Metric['predictions']
    del Metric['targets']
    return {
        "Model":task, 
        "Prediction_df": data,  # only return test_dataset
        "Metric":Metric
    }
