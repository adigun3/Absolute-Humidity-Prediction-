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

This dataset is split into independent variables "X", containing all the varibles in the dataset with the exception of the predicted absolute humidity which is the dependent varriable "y".

Data Wrangling
--------------
After importing the dataset into google colab, the next step is to transform it into the right format by encoding the dataset into categorical data using the OneHotEncoder library and then converting the independent variable X into a data array format. Later on, the data was splited into the Training set (X_train, y_train) and Test set (X_test and y_test).


## Encoding categorical data

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


Training the Dataset
--------------
After the dataset has been separated into the Traing and Test set, I decided to use the Multple Linear Regression Model on the Training set to train the dataset. I used the LinearRegression library from sklearn.Linear_model to create a regressor which I applied as "fit" on the X_train and y_train (independent variables).

Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

Making the prediction of Absolute Humidity
--------------

After creating the regressor, the regressor was applied as a predict on the X_test to compute a y_pred which is the predicted "y" values. The y_pred represent the predicted dependent variable from the independent test dataset (X_test) and this prediction was compared to y_test, the dependent test variables.


Predicting the Test set results 

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

[[ 4.52e-01  6.82e-01]

 [ 2.76e+00  4.13e-01]
 
 [ 2.20e+00  1.16e+00]
 
 [-1.99e+02 -2.00e+02]
 
 [-1.09e-02  1.00e+00]
 
 [ 1.59e+00  1.16e+00]]


