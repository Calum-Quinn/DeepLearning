import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Create the dataset
pizza_types = pd.read_csv('pizza_types.csv', index_col=0)

# Split the data into training and test
training_dataset = pizza_types.sample(frac=0.8)
testing_dataset = pizza_types[~pizza_types.index.isin(training_dataset.index)]

#%%
# Create the neural network
pizza_type_model = Sequential()
pizza_type_model.add(Dense(3, input_dim=15, activation='softmax'))

# Configure the model for training
sgd = SGD()
pizza_type_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on the training set while using 20% for validation
history_sgd_pizza_type_model = pizza_type_model.fit(
    training_dataset[['corn', 'olives', 'mushrooms', 'spinach', 'pineapple', 'artichoke', 'chilli', 'pepper', 'onion',
                      'mozzarella', 'egg', 'pepperoni', 'beef', 'chicken', 'bacon',]],
    training_dataset[['vegan', 'vegetarian', 'meaty']],
    epochs=200,
    validation_split=0.2,
)

#%%
# Create the neural network
pizza_type_model = Sequential()
pizza_type_model.add(Dense(3, input_dim=15, activation='softmax'))

# Configure the model for training
adam = Adam()
pizza_type_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model on the training set while using 20% for validation
history_adam_pizza_type_model = pizza_type_model.fit(
    training_dataset[['corn', 'olives', 'mushrooms', 'spinach', 'pineapple', 'artichoke', 'chilli', 'pepper', 'onion',
                      'mozzarella', 'egg', 'pepperoni', 'beef', 'chicken', 'bacon',]],
    training_dataset[['vegan', 'vegetarian', 'meaty']],
    epochs=200,
    validation_split=0.2,
)

#%%
# Test the model on the test est after having trained it
test_loss, test_acc = pizza_type_model.evaluate(
    testing_dataset[['corn', 'olives', 'mushrooms', 'spinach', 'pineapple', 'artichoke', 'chilli', 'pepper', 'onion',
                     'mozzarella', 'egg', 'pepperoni', 'beef', 'chicken', 'bacon',]],
    testing_dataset[['vegan', 'vegetarian', 'meaty']]
)

print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")

#%%
# Compare the efficiency of sgd (stochastic gradient descent) and adam (adaptive momentum estimation)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes = axes.flatten()

axes[0].plot(history_sgd_pizza_type_model.history['loss'])
axes[0].plot(history_sgd_pizza_type_model.history['accuracy'])
axes[0].set_title('SGD based Model Training History')
axes[0].set_ylabel('Value')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Loss', 'Accuracy'], loc='center right')
axes[1].plot(history_adam_pizza_type_model.history['loss'])
axes[1].plot(history_adam_pizza_type_model.history['accuracy'])
axes[1].set_title('Adam based Model Training History')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Loss', 'Accuracy'], loc='center right')
plt.show()
