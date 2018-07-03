
### Import modules 


```python
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Input, Dense, TimeDistributed, Embedding, Bidirectional, RepeatVector
from keras.models import Model, Sequential
from keras.layers import Activation, LSTM
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

```

### Load data


```python
def load_file(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding = "utf8") as f:
        data = f.read().split('\n')

    return data
```


```python
english_sentences = load_file('data/en_data')
french_sentences = load_file('data/fr_data')

print('Dataset Loaded')
```

    Dataset Loaded



```python
print(english_sentences[0])
print(french_sentences[0])

```

    paris is sometimes warm during june , but it is usually hot in july .
    paris est parfois chaud en juin , mais il est généralement chaud en juillet .


### Tokenize


```python
def tokenize(input):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input)
    input_tokenized = tokenizer.texts_to_sequences(input)
    
    return input_tokenized, tokenizer

french_data_tokenized, french_tokenizer = tokenize(french_sentences)
english_data_tokenized, english_tokenizer = tokenize(english_sentences)

print(english_data_tokenized[1])
print(french_data_tokenized[1])
```

    [18, 1, 10, 68, 4, 46, 7, 3, 1, 9, 64, 2, 37]
    [29, 1, 9, 125, 37, 11, 46, 6, 3, 1, 12, 58, 2, 44]


### Add Padding


```python
def pad(input, length=None):
   
    if length == None:
        length = max([len(seq) for seq in input])
        
    return pad_sequences(input, maxlen=length, padding='post')


french_data_padded = pad(french_data_tokenized)
french_data_padded = french_data_padded.reshape(*french_data_padded.shape, 1)
english_data_padded = pad(english_data_tokenized)

print(english_data_padded[1])
print(french_data_padded[1])
```

    [18  1 10 68  4 46  7  3  1  9 64  2 37  0  0]
    [[ 29]
     [  1]
     [  9]
     [125]
     [ 37]
     [ 11]
     [ 46]
     [  6]
     [  3]
     [  1]
     [ 12]
     [ 58]
     [  2]
     [ 44]
     [  0]
     [  0]
     [  0]
     [  0]
     [  0]
     [  0]
     [  0]]


### Build a simple RNN model


```python
def simple_model(input_shape, output_len, num_uniq_fr_words):

    model = Sequential()
    model.add(GRU(units=256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(num_uniq_fr_words)))
    model.add(Activation('softmax'))
    
    learning_rate = 0.002
        
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model
```

### Advanced RNN model


```python
def advanced_model(input_shape, output_len, num_uniq_en_words, num_uniq_fr_words):
    
    model = Sequential()
    model.add(Embedding(num_uniq_en_words, 512, input_length=input_shape[1]))
    model.add(Bidirectional(LSTM(512, return_sequences=False)))
    model.add(RepeatVector(output_len))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_uniq_fr_words)))
    model.add(Activation('softmax'))
    
    learning_rate=0.002
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
```

### Train the model


```python
model = advanced_model(english_data_padded.shape, french_data_padded.shape[1], 
                       len(english_tokenizer.word_index), len(french_tokenizer.word_index))
model.fit(english_data_padded, french_data_padded, batch_size=1024, epochs=10, validation_split=0.2)
```

    Train on 110288 samples, validate on 27573 samples
    Epoch 1/10
    110288/110288 [==============================] - 49s - loss: 2.0923 - acc: 0.5273 - val_loss: nan - val_acc: 0.6412
    Epoch 2/10
    110288/110288 [==============================] - 47s - loss: 1.0994 - acc: 0.6960 - val_loss: nan - val_acc: 0.7499
    Epoch 3/10
    110288/110288 [==============================] - 47s - loss: 0.6902 - acc: 0.7965 - val_loss: nan - val_acc: 0.8418
    Epoch 4/10
    110288/110288 [==============================] - 47s - loss: 0.3837 - acc: 0.8880 - val_loss: nan - val_acc: 0.9144
    Epoch 5/10
    110288/110288 [==============================] - 48s - loss: 0.2233 - acc: 0.9351 - val_loss: nan - val_acc: 0.9502
    Epoch 6/10
    110288/110288 [==============================] - 48s - loss: 0.1509 - acc: 0.9559 - val_loss: nan - val_acc: 0.9551
    Epoch 7/10
    110288/110288 [==============================] - 48s - loss: 0.1178 - acc: 0.9651 - val_loss: nan - val_acc: 0.9646
    Epoch 8/10
    110288/110288 [==============================] - 48s - loss: 0.0935 - acc: 0.9725 - val_loss: nan - val_acc: 0.9685
    Epoch 9/10
    110288/110288 [==============================] - 48s - loss: 0.0803 - acc: 0.9763 - val_loss: nan - val_acc: 0.9739
    Epoch 10/10
    110288/110288 [==============================] - 48s - loss: 0.0691 - acc: 0.9794 - val_loss: nan - val_acc: 0.9735





    <keras.callbacks.History at 0x21fac003a20>




```python
    fr_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
    fr_id_to_word[0] = '|empty space|'

    sentence = 'china is usually hot during february and it is usually wonderful in winter'
    sentence = [english_tokenizer.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=english_data_padded.shape[-1], padding='post')

    sentences = np.array([sentence[0]])
    predictions = model.predict(sentence)


    print(' '.join([fr_id_to_word[np.argmax(x)] for x in predictions[0]]))


```

    chine est généralement chaud en février et il est généralement merveilleux en hiver |empty space| |empty space| |empty space| |empty space| |empty space| |empty space| |empty space| |empty space|

