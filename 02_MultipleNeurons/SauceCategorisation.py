import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

#%%
# Create the dataset
bad_pizza_dataset = pd.DataFrame.from_dict({
    'tomato_sauce': ['no', 'no', 'yes', 'yes'],
    'barbecue_sauce': ['no', 'yes', 'no', 'yes'],
    'result': ['sauce error', 'good', 'good', 'sauce error']
}
)

# Convert the data to machine interpretable numbers
bad_pizza_dataset['c_tomato_sauce'] = bad_pizza_dataset['tomato_sauce'].apply(lambda x: 1 if x == 'yes' else 0)
bad_pizza_dataset['c_barbecue_sauce'] = bad_pizza_dataset['barbecue_sauce'].apply(lambda x: 1 if x == 'yes' else 0)
bad_pizza_dataset['c_result'] = bad_pizza_dataset['result'].apply(lambda x: 1 if x == 'sauce error' else 0)

# Define the neural network
input_layer = Dense(units=2, input_dim=2, activation='sigmoid')
output_layer = Dense(units=1, activation='sigmoid')

# Create the neural network
bad_pizza_model = Sequential()
bad_pizza_model.add(input_layer)
bad_pizza_model.add(output_layer)

#%%
# Configure the model for training
sgd = SGD()
bad_pizza_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = bad_pizza_model.fit(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result'],
    epochs=3000,
)

# Final evaluation after training
test_loss, test_acc = bad_pizza_model.evaluate(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result']
)

print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")

#%%
# Graph showing the evolution of the loss and accuracy according to the epoch
plt.figure(figsize=(7, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model Training History')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='center right')
plt.show()

#%%
# Configure the model for training with a high learning rate
high_lr_sgd = SGD(learning_rate=10)
bad_pizza_model.compile(loss='binary_crossentropy', optimizer=high_lr_sgd, metrics=['accuracy'])

# Train the model
high_lr_model_history = bad_pizza_model.fit(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result'],
    epochs=3000
)

# Final evaluation after training
test_loss, test_acc = bad_pizza_model.evaluate(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result']
)

print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")

#%%
# Graph showing the evolution of the loss and accuracy according to the epoch
plt.figure(figsize=(7, 6))
plt.plot(high_lr_model_history.history['loss'])
plt.plot(high_lr_model_history.history['accuracy'])
plt.title('Model Training History')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='center right')
plt.show()

#%%
# Configure the model for training with a medium learning rate
medium_lr_sgd = SGD(learning_rate=1)
bad_pizza_model.compile(loss='binary_crossentropy', optimizer=medium_lr_sgd, metrics=['accuracy'])

# Train the model
medium_lr_model_history = bad_pizza_model.fit(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result'],
    epochs=3000
)

# Final evaluation after training
test_loss, test_acc = bad_pizza_model.evaluate(
    bad_pizza_dataset[['c_tomato_sauce', 'c_barbecue_sauce']],
    bad_pizza_dataset['c_result']
)

print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")

#%%
# Graph showing the evolution of the loss and accuracy according to the epoch
plt.figure(figsize=(7, 6))
plt.plot(medium_lr_model_history.history['loss'])
plt.plot(medium_lr_model_history.history['accuracy'])
plt.title('Model Training History')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='center right')
plt.show()
