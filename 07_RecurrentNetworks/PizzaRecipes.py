import numpy as np

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
        batched_x.append(np.asarray(x[idx:idx + batch_size]))
        batched_y.append(np.asarray(y[idx:idx + batch_size]))


