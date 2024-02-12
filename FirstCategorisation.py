import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

corn_and_olives_dataset = pd.DataFrame.from_dict({
    'shape': ['round', 'oval'],
    'color': ['yellow', 'green'],
    'ingredient_type': ['corn', 'olives']
}
)

corn_and_olives_dataset['c_shape'] = corn_and_olives_dataset['shape'].apply(lambda x: 1 if x == 'round' else 0)
corn_and_olives_dataset['c_color'] = corn_and_olives_dataset['color'].apply(lambda x: 1 if x == 'yellow' else 0)
corn_and_olives_dataset['c_ingredient_type'] = corn_and_olives_dataset['ingredient_type'].apply(lambda x: 1 if x == 'corn' else 0)

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

single_neuron_layer = Dense(
    units=1,
    input_dim=2,
    activation='sigmoid'
)

loss='binary_crossentropy'

single_neuron_model = Sequential()
sgd = SGD()

single_neuron_model.add(single_neuron_layer)
single_neuron_model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
single_neuron_model.summary()

history = single_neuron_model.fit(
    corn_and_olives_dataset[['c_shape', 'c_color']].values,
    corn_and_olives_dataset[['c_ingredient_type']].values,
    epochs=2500)

test_loss, test_acc = single_neuron_model.evaluate(
    corn_and_olives_dataset[['c_shape', 'c_color']],
    corn_and_olives_dataset['c_ingredient_type']
)

print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")


#%%
