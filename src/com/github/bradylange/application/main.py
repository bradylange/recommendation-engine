# Developer: Brady Lange
# Date: 08/18/2019
# Description:

# Import requires libraries
import pandas as pd
import numpy as np
import os
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and Explore Data
# =============================================================================
# Load craft beer dataset
beers = pd.read_csv(r"..\..\..\..\..\data\beers.csv",dtype = {"brewery_id": str})
# Remove index column
beers = beers.iloc[:, 1:]

# Explore the dataset
print(beers.head())
print(beers.tail())
print(beers.describe())
print(beers.shape)
print(beers.size)
print(len(beers))
print(beers.columns)
print("\nNull Values:\n" + str(beers.isnull().sum()))

# Preprocess Data
# =============================================================================
beersId = beers[["id", "brewery_id"]]
beers.drop(beersId.columns, inplace = True, axis = 1)
imputerNum = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputerNum.fit(beers[["abv", "ibu"]])
beers[["abv", "ibu"]] = imputerNum.transform(beers[["abv", "ibu"]])
imputerCat = SimpleImputer(missing_values = np.nan, strategy = "constant",
                           fill_value = "null")
imputerCat.fit(beers["style"].values.reshape(-1, 1))
beers["style"] = imputerCat.transform(beers["style"].values.reshape(-1, 1))
print("\nNull Values:\n" + str(beers.isnull().sum()))

std = StandardScaler()
beers[["abv", "ibu", "ounces"]] = std.fit_transform(beers[["abv", "ibu", "ounces"]])

print(beers.corr())
# One Hot Encode, drop first encoded column to prevent multicollinearity
beers = pd.get_dummies(beers, drop_first = True)

# Autoencoder - Dimensionality Reduction
# =============================================================================
inputDim = beers.shape[1]
codeDim = 32

inputLayer = Input(shape = (inputDim, ), name = "input_layer")
# Encoder layers
encoded = Dense(512, activation = "relu", name = "encoded_hl_1")(inputLayer)
encoded = Dense(256, activation = "relu", name = "encoded_hl_2")(encoded)
encoded = Dense(128, activation = "relu", name = "encoded_hl_3")(encoded)
# Code layer
encoded = Dense(codeDim, activation = "relu", name = "code_hl")(encoded)
# Decoder layers
decoded = Dense(128, activation = "relu", name = "decoded_hl_1")(encoded)
decoded = Dense(256, activation = "relu", name = "decoded_hl_2")(decoded)
decoded = Dense(512, activation = "relu", name = "decoded_hl_3")(decoded)
decoded = Dense(inputDim, activation = "sigmoid", name = "output_layer")(decoded)

# Instantiate Autoencoder model
autoencoder = Model(inputLayer, decoded)

# Instantiate encoder model
encoder = Model(inputLayer, encoded)

# Instantiate decoder model
codedInput = Input(shape = (codeDim, ))
decoder = autoencoder.layers[-4](codedInput)
decoder = autoencoder.layers[-3](decoder)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(codedInput, decoder)

# Configure Autoencoder model
autoencoder.compile(optimizer = "sgd", loss = "mean_squared_error",
                    metrics = ["accuracy"])

modCheck = ModelCheckpoint(r"models\checkpoints\weights_{epoch:02d}_{val_loss:.2f}.hdf5",
                           mode = "min")
erlyStop = EarlyStopping(monitor = "val_loss", patience = 2)
# Train Autoencoder model
autoencoder.fit(beers, beers,
                epochs = 10,
                batch_size = 128,
                shuffle = True)
                #callbacks = [modCheck, erlyStop])

# Encode/Decode Data
# =============================================================================
encodedBeers = encoder.predict(beers)
decodedBeers = decoder.predict(encodedBeers)

# K-Means Clustering
# =============================================================================
km = KMeans(n_clusters = 3)
km.fit(encodedBeers)
km.predict(encodedBeers)

# Recommendation Engine
# =============================================================================