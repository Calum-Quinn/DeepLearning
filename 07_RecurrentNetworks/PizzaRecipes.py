import numpy as np
import os
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import train
from tensorflow import TensorShape

# Create the array
data_pizza = np.load("recipes/data_pizza.npy", allow_pickle=True)

# Filter out the useless information
ingredients_only = [t['ingredients'] for t in data_pizza]

# Create a string for each recipe's ingredients
joined_ingredients = []
for set_of_ingredients in ingredients_only:
    str_ingredients = ''
    for ingredient in set_of_ingredients:
        str_ingredients += ' '.join([str(t) for t in ingredient]) + ' '
    joined_ingredients.append(str_ingredients.replace('  ', ' '))

text = ' '.join(joined_ingredients).lower()

# Assign a number to each character
vocabulary = sorted(set(text))
character_to_number = {}
for idx, character in enumerate(vocabulary):
    character_to_number[character] = idx
number_to_character = np.array(vocabulary)

text_as_numbers = [character_to_number[character] for character in text]

# Separate the text into chunks
chunk_length = 15
chunk_length_with_extra_character = chunk_length + 1
chunks = []
for idx in range(0, len(text_as_numbers), chunk_length_with_extra_character):
    chunks.append(text_as_numbers[idx:idx + chunk_length_with_extra_character])

''' Check the chunks
for number_chunk in chunks[:5]:
    text_chunk = [number_to_character[item] for item in number_chunk]
    print(f"Number sequence: {number_chunk}")
    print(f"As text: {''.join(text_chunk)}")
    print('')
'''

# Convert all the chunks into input and target chunks
x = []
y = []
for chunk in chunks:
    x.append(chunk[:-1])
    y.append(chunk[1:])

# Create batches of these chunks
batch_size = 64
batched_x = []
batched_y = []
for idx in range(0, min(len(x), len(y)), batch_size):
    if (
            len(x[idx:idx + batch_size]) == batch_size and
            len(y[idx:idx + batch_size]) == batch_size
    ):
        batch_x = np.asarray(x[idx:idx + batch_size])
        batch_y = np.asarray(y[idx:idx + batch_size])
        batched_x.append(batch_x)
        batched_y.append(batch_y)

# Concatenate all batches into one large batch
batched_x = np.concatenate(batched_x)
batched_y = np.concatenate(batched_y)


# Define and build the model
def create_model(batch_size):
    input_layer = Embedding(
        input_dim=len(vocabulary),
        output_dim=256,
        batch_input_shape=[batch_size, None]
    )

    hidden_layer = LSTM(
        units=256,
        return_sequences=True,
        stateful=True
    )

    output_layer = Dense(units=len(vocabulary), activation='softmax')

    rnn_model = Sequential([
        input_layer,
        hidden_layer,
        output_layer,
    ])

    return rnn_model


# Create the model
rnn_model = create_model(batch_size)
rnn_model.summary()

rnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Set up the storage during training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Train the model
history = rnn_model.fit(
    batched_x,
    batched_y,
    epochs=200,
    callbacks=[checkpoint_callback],
    batch_size=batch_size
)

# New version of the model which takes 1 batch of 1 character
latest_checkpoint = train.latest_checkpoint(checkpoint_dir)
single_input_rnn_model = create_model(batch_size=1)
single_input_rnn_model.load_weights(latest_checkpoint)
single_input_rnn_model.build(TensorShape([1, None]))

# Determine the average length of a list of ingredients and round it up to an integer
average_length_ingredients = np.mean([len(t) for t in joined_ingredients])
output_sequence_length = int(np.round(average_length_ingredients))
print(output_sequence_length)

# Choose a starting character and use the model to predict what the next character will be
starting_character = 'a'

model_input = [[character_to_number[s] for s in starting_character]]
generated_text = []
single_input_rnn_model.reset_states()
for i in range(output_sequence_length):
    predictions = single_input_rnn_model.predict(model_input)
    predicted_id = np.argmax(predictions)
    model_input = np.array([np.array([predicted_id])])
    generated_text.append(number_to_character[predicted_id])

print(starting_character + ''.join(generated_text))
