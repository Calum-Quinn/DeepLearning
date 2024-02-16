import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Create dataframe and make it machine interpretable
traffic_data = pd.read_csv("traffic_data.csv")
traffic_data['c_type'] = traffic_data['type'].apply(lambda x: 1 if x == 'traffic' else 0)

# Convert days to new columns
traffic_data = traffic_data.join(pd.get_dummies(traffic_data['day']))

# Convert boolean columns to numerical values
bool_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
traffic_data[bool_columns] = traffic_data[bool_columns].astype(int)

# Separate the data into training and test sets
training_dataset = traffic_data.sample(frac=0.8)
testing_dataset = traffic_data[~traffic_data.index.isin(training_dataset.index)]

# Select the columns we'll be using
input_columns = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
    'hour',
    'minute',
    'second'
]
output_column = training_dataset['c_type']

#%%
# Define the model
input_layer = Dense(units=50, input_dim=len(input_columns), activation='relu')
hidden_layer = Dense(units=50, activation='relu')
output_layer = Dense(units=1, activation='sigmoid')

# Create the neural network
traffic_model = Sequential([
    input_layer, Dropout(0.1),
    hidden_layer, Dropout(0.1),
    output_layer
])

# Configure the model for training
adam = Adam()
traffic_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model on the training set while using 10% for validation
history_model = traffic_model.fit(
    training_dataset[input_columns],
    output_column,
    epochs=30,
    validation_split=0.1,
    batch_size=100
)
