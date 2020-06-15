# Absolute-Humidity-Prediction

Introduction
------------
Predicting Absolute Humidity from Air Quality dataset from responses of a gas multisensor device deploy on field from an Italian city using artificial neural network (ANN) and Multiple regression model to determine the best predictive result.   
I am providing a documentation to show I built and created the python algorithms and tools to make the prediction from both Artificial Neural Network and Multiple regression models and arrive with the best predictive result. The workflow is shown below. 

Getting Data
------------

To start with, the dataset used in the workflow (Air Quality Data Set) was accessed and downloaded from UCI Machine Learning Resiporatory. It contain 9358 instances of hourly average responses from a device located on a field in a highly polluted area in Italy. The dataset consist of fouteen attributes ranging from True hourly averaged concentration CO, True hourly averaged Benzene concentration, Temperature, Relative Humidity, Absolute Humidity anong others.
The dataset is named AirQualityUCI.csv

Import data to project:

dataset = pd.read_csv('AirQualityUCI.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
