
# LSTM-HVAC
## Devoted to the use of LSTM to predict daily HVAC consumption in buildings.
## Use of the code:
### 1. Load dataset:
Store the dataset in the data directory and create a python class that inherits from the 
TimeSeriesDataset class as follows:
```python
class DatasetHVAC(TimeSeriesDataset):
    def __init__(self, *args,**kwargs):
        self.dir = 'hvac_pred/data/hvac/dataHVAC.npy'
        data = np.load(self.dir, allow_pickle=True)[:, 2:].astype('float')
        super(DatasetHVAC, self).__init__(data, 4, *args, **kwargs)
        del data
```
The path to the data and the column index of the target time series to be forecasted must be specified. 
The data to be loaded must be ordered in time. Additionally, variables such as the time and date can be 
removed (in the previous code, the first 2 columns were removed).

### 2. Create the settings JSON file that stores the hyperparameters to be used. 
The configuration file must be stored in the Settings directory. Some examples are already provided as a template.
```json
{
	"chk_file" : "mspm2_checkpoint",
	"device" : "cuda:0",
	"epochs" : 100, 
	"samples" : 100000, 
	"batch_size" : 100,
	"timesteps" : 60,
	"horizon" : 96,
	"output_size": 1,
	"predictor" : "MSPM2",
	"learning_rate" : 1e-3,
	"spl" : [0.8, 0.2, 0], 
	"resume" : 0,
	"ann_units" : [30, 30]
}
```
The hyper-parameters have the following meaning:
- chk_file: Name of the file where the model checkpoint will be stored.
- devide: Device used to train the gpu.
- epochs: Number of epochs during training of the predictor.
- batch_size: The batch size to be used only during training. It is implemented in a way that preserves the dynamics of the models.
- timesteps: The number of past timesteps (tau) to be unrolled in the BPTT.
- horizon: Horizon of the predictions. If the predictor forecasts multiple step at once (i.e. from k+1 to k+96), then horizon refers to 
the first estimation. It is combined with output_size parameter to deal with multi-step predictions.
- output_size: It determines how many predictions are performed at once at each time step. Therefore, for multi-step predictors it has to 
be combined with horizon.
- predictor: RNN architecture to be trained. Although 3 LSTM based predictors are implemented, new architectures can be included by creating the 
corresponding class that extends BasePredictor class. Additionally, the python dictionary in the main file must be modified as well.
- learning rate: Learning rate to be used by Adam optimizer.
- spl: indicates the proportions of the data corresponding to train, test and validation subsets respectively.
- resume: restores the training process from checkpoint. As it is also implemented as a command line argument, ignore this parameter.
- ann_units: number of LSTM units.

### 3. Run the program as follows:
- Train from zero: ```python main.py --json mspm1```
- Restore training from checkpoint: ```python main.py --json mspm1 --resume```
- Evaluate trained model (checkpoint has to be stored): ```python main.py --json mspm1 --predict```


