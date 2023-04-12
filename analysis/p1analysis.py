# %%
import tensorflow as tf


# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("/Users/paulhwang/Desktop/Distributed Computing/Project1/finaldata.csv")

# %%
df

# %%
df_comb = df.copy()
df_comb['Combined_Tag'] = df['tag1'] + ',' + df['tag2'] + ',' + df['tag3']
df_comb.drop(['tag1', 'tag2', 'tag3'], axis=1, inplace=True)

# %%
df_comb

# %%
df_clean = df_comb[["author","total_posts", "avg_posts_year", "avg_word_count", "avg_paragraph_count", "Combined_Tag", "log_avg_rating"]]

df_clean = df_comb[["total_posts", "avg_posts_year", "avg_word_count", "avg_paragraph_count", "log_avg_rating"]]

df_clean = df_comb[["total_posts", "avg_posts_year", "avg_word_count", "avg_paragraph_count", "avg_rating"]]

# %%
df_clean

# %%
X = df_clean.iloc[:,0:4]
y = df_clean.iloc[:,4]

# %%
X.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# tensorflow

# %%
X_train.shape

# %%
sum(np.isnan(y_train))

# %% [markdown]
# start neural net

# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(4,))) # Add input layer with input shape (4,)
model.add(tf.keras.layers.Dense(units=16, activation='relu')) # Add first dense layer with 16 units and ReLU activation
model.add(tf.keras.layers.Dropout(0.03))
model.add(tf.keras.layers.Dense(units=32, activation='relu')) # Add second dense layer with 32 units and ReLU activation
model.add(tf.keras.layers.Dropout(0.03))
model.add(tf.keras.layers.Dense(units=1)) # Add output layer with 1 unit (for regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics = "accuracy") # Use Adam optimizer and mean squared error (MSE) as the loss

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100) # Fit the model to the data for 100 epochs with batch size of 32


# %%
np.sqrt(20837184)
np.sqrt(24034400)

# %%
print(history.params)

# %%
results = model.evaluate(X_test_scaled, y_test)

# %% [markdown]
# catboost

# %%
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# %%
df_clean

# %%
X = df_clean.iloc[:,:-1]
y = df_clean.iloc[:,-1:]

## split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X.shape

# %%
catreg = CatBoostRegressor(random_state=42, subsample=1)
catreg.fit(X_train_scaled, y_train)

# %%
catreg = CatBoostRegressor(random_state=42)

lr = np.arange(0.01, 0.05, 0.01)
l2_reg = [1, 3, 10, 50]
params_cat = {"learning_rate": lr, "depth": range(4, 10), "l2_leaf_reg": l2_reg}

gs_cat = GridSearchCV(
    catreg, param_grid=params_cat, scoring="neg_mean_squared_error", cv=6
)

gs_cat.fit(X_train_scaled, y_train)

y_pred_test = gs_cat.predict(X_test_scaled)

RMSE = mean_squared_error(y_test, y_pred_test, squared=False)

## should return 3.845808036788226
print("RMSE for catbosting is :", str(RMSE))

## should return 'depth': 5, 'l2_leaf_reg': 0.5, 'learning_rate': 0.045
print("Best parameters for catboosting are as follows: ", str(gs_cat.best_params_))


# %%
X_train

# %%
np.min(X_train.iloc[:,0])

# %%
y_train

# %% [markdown]
# prepping for r things

# %%
## tot_post
print("Tot Post Min:", np.min(X_train.iloc[:,0]))
print("Tot Post Low:", np.percentile(X_train.iloc[:,0], 25))
print("Tot Post Med:", np.median(X_train.iloc[:,0]))
print("Tot Post High:", np.percentile(X_train.iloc[:,0], 75))
print("Tot Post Max:", np.max(X_train.iloc[:,0]))

## avg_post
print("Tot Avg Post Min:", np.min(X_train.iloc[:,1]))
print("Tot Avg Post Low:", np.percentile(X_train.iloc[:,1], 25))
print("Tot Avg Post Med:", np.median(X_train.iloc[:,1]))
print("Tot Avg Post High:", np.percentile(X_train.iloc[:,1], 75))
print("Tot Avg Post Max:", np.max(X_train.iloc[:,1]))

## avg_word
print("Tot Word Min:", np.min(X_train.iloc[:,2]))
print("Tot Word Low:", np.percentile(X_train.iloc[:,2], 25))
print("Tot Word Med:", np.median(X_train.iloc[:,2]))
print("Tot Word High:", np.percentile(X_train.iloc[:,2], 75))
print("Tot Word Max:", np.max(X_train.iloc[:,2]))

## tot_para
print("Tot Para Min:", np.min(X_train.iloc[:,3]))
print("Tot Para Low:", np.percentile(X_train.iloc[:,3], 25))
print("Tot Para Med:", np.median(X_train.iloc[:,3]))
print("Tot Para High:", np.percentile(X_train.iloc[:,3], 75))
print("Tot Para Max:", np.max(X_train.iloc[:,3]))

## ratings
print("Rating Min:", np.min(y_train.iloc[:,0]))
print("Rating Low:", np.percentile(y_train.iloc[:,0], 25))
print("Rating Med:", np.median(y_train.iloc[:,0]))
print("Rating High:", np.percentile(y_train.iloc[:,0], 75))
print("Rating Max:", np.max(y_train.iloc[:,0]))

# %%
df1 = pd.DataFrame({"tot_posts": [1,1,1,3,366]})
df2 = pd.DataFrame({"avg_posts": [1,1,1,2,31.75]})
df3 = pd.DataFrame({"tot_words": [0,810,1562,3130,503717]})
df4 = pd.DataFrame({"tot_para": [1,1,1,4/3,190]})

# %%
from itertools import product

# %%
vec1 = [1,1,1,3,366]
vec2 = [1,1,1,2,31.75]
vec3 = [0,810,1562,3130,503717]
vec4 = [1,1,1,4/3,190]

# %%
vec1 = [1,3,366]
vec2 = [1,2,31.75]
vec3 = [0,810,1562,3130,503717]
vec4 = [1,4/3,190]

# %%
combinations = list(product(vec1, vec2, vec3, vec4))

df_fin = pd.DataFrame(combinations, columns=['total_posts', 'avg_posts_year', 'avg_word_count', 'avg_paragraph_count'])


# %%
X_train.columns

# %%
print(df_fin)

# %%
X_gen_scaled = scaler.transform(df_fin)

# %%
y_pred_test = gs_cat.predict(X_gen_scaled)

# %%
y_pred_test

# %%
df_fin["pred_rating"] = y_pred_test

# %%
df_fin

# %%
print(df_fin)

# %%
df_fin.to_csv('predictions.csv', index=False)

# %% [markdown]
# outlier removed catboost

# %%
pd.options.display.max_rows = 10

# %%
df_clean

# %%
Q1 = df_clean["avg_rating"].quantile(0.25)
Q3 = df_clean["avg_rating"].quantile(0.75)

# %%
IQR = Q3 - Q1
k = 1.5
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR

# %%
df_removed = df_clean[(df_clean["avg_rating"] >= lower_bound) & (df_clean["avg_rating"] <= upper_bound)]

# %%
df_removed.shape

# %%
X = df_removed.iloc[:,:-1]
y = df_removed.iloc[:,-1:]

## split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
catreg = CatBoostRegressor(random_state=42)

lr = np.arange(0.01, 0.05, 0.01)
l2_reg = [1, 3, 10, 50]
params_cat = {"learning_rate": lr, "depth": range(4, 10), "l2_leaf_reg": l2_reg}

gs_cat = GridSearchCV(
    catreg, param_grid=params_cat, scoring="neg_mean_squared_error", cv=6
)

gs_cat.fit(X_train_scaled, y_train)

# %%
y_pred_test = gs_cat.predict(X_test_scaled)

RMSE = mean_squared_error(y_test, y_pred_test, squared=False)

## should return 3.845808036788226
print("RMSE for catbosting is :", str(RMSE))

## should return 'depth': 5, 'l2_leaf_reg': 0.5, 'learning_rate': 0.045
print("Best parameters for catboosting are as follows: ", str(gs_cat.best_params_))

# %%
## tot_post
print("Tot Post Min:", np.min(X_test.iloc[:,0]))
print("Tot Post Low:", np.percentile(X_test.iloc[:,0], 25))
print("Tot Post Med:", np.median(X_test.iloc[:,0]))
print("Tot Post High:", np.percentile(X_test.iloc[:,0], 75))
print("Tot Post Max:", np.max(X_test.iloc[:,0]))

## avg_post
print("Tot Avg Post Min:", np.min(X_test.iloc[:,1]))
print("Tot Avg Post Low:", np.percentile(X_test.iloc[:,1], 25))
print("Tot Avg Post Med:", np.median(X_test.iloc[:,1]))
print("Tot Avg Post High:", np.percentile(X_test.iloc[:,1], 75))
print("Tot Avg Post Max:", np.max(X_test.iloc[:,1]))

## avg_word
print("Tot Word Min:", np.min(X_test.iloc[:,2]))
print("Tot Word Low:", np.percentile(X_test.iloc[:,2], 25))
print("Tot Word Med:", np.median(X_test.iloc[:,2]))
print("Tot Word High:", np.percentile(X_test.iloc[:,2], 75))
print("Tot Word Max:", np.max(X_test.iloc[:,2]))

## tot_para
print("Tot Para Min:", np.min(X_test.iloc[:,3]))
print("Tot Para Low:", np.percentile(X_test.iloc[:,3], 25))
print("Tot Para Med:", np.median(X_test.iloc[:,3]))
print("Tot Para High:", np.percentile(X_test.iloc[:,3], 75))
print("Tot Para Max:", np.max(X_test.iloc[:,3]))

## ratings
print("Rating Min:", np.min(y_test.iloc[:,0]))
print("Rating Low:", np.percentile(y_test.iloc[:,0], 25))
print("Rating Med:", np.median(y_test.iloc[:,0]))
print("Rating High:", np.percentile(y_test.iloc[:,0], 75))
print("Rating Max:", np.max(y_test.iloc[:,0]))

# %%
vec1 = [1,1,1,2,90]
vec2 = [1,1,1,1.8,22.3]
vec3 = [0,750,1423,2707,158788]
vec4 = [1,1,1,1.428,55]

# %%
vec1 = [1,1,1,2,90]
vec2 = [1,1,1,1.8,22.3]
vec3 = [0,750,1423,2707,158788]
vec4 = [1,1,1,1.428,55]

# %%
combinations = list(product(vec1, vec2, vec3, vec4))

df_fin = pd.DataFrame(combinations, columns=['total_posts', 'avg_posts_year', 'avg_word_count', 'avg_paragraph_count'])


# %%
X_gen_scaled = scaler.transform(df_fin)

# %%
y_pred_test = gs_cat.predict(X_gen_scaled)

# %%
df_fin["pred_rating"] = y_pred_test

# %%
df_fin

# %%
df_fin.to_csv('predictions_rm.csv', index=False)

# %% [markdown]
# tensorflow with outlier removed

# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(4,))) # Add input layer with input shape (4,)
model.add(tf.keras.layers.Dense(units=16, activation='relu')) # Add first dense layer with 16 units and ReLU activation
model.add(tf.keras.layers.Dropout(0.03))
model.add(tf.keras.layers.Dense(units=32, activation='relu')) # Add second dense layer with 32 units and ReLU activation
model.add(tf.keras.layers.Dropout(0.03))
model.add(tf.keras.layers.Dense(units=1)) # Add output layer with 1 unit (for regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics = "accuracy") # Use Adam optimizer and mean squared error (MSE) as the loss

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1) # Fit the model to the data for 100 epochs with batch size of 32


# %%
np.sqrt(386942.9688)

# %%
print(history.params)

# %%
results = model.evaluate(X_test_scaled, y_test)

# %%
np.sqrt(374817.5)

# %%
y_pred_nn = model.predict(X_gen_scaled)

# %%
y_pred_nn


