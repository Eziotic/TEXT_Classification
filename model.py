from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Loaddata import x_tr, y_tr, x_va, y_va, vocab

print('Loading data')

x_train = x_tr
y_train = y_tr
x_valuation = x_va
y_valuation = y_va

embedding_dim = 300
sequence_length = 128
filter_sizes = [3,4,5]
num_filters = 128
drop = 0.5
vocabulary_size = vocab
epochs = 10
batch_size = 50

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=1, activation='sigmoid')(dropout)


model = Model(inputs=inputs, outputs=output)


early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[early_stopping],
          validation_data=(x_valuation, y_valuation)
          )
json_file = model.to_json()
open("train_model.json","w").write(json_file)
model.save_weights("train_model.h5")

y_pred = model.predict(x_valuation)
y_pred_list = []
for i in range(len(y_pred)):
    if(y_pred[i] > 0.5):
        y_pred_list.append(1)
    else:
        y_pred_list.append(0)


confusion = confusion_matrix(y_valuation,y_pred_list)
print(confusion)
tn, fp, fn, tp = confusion_matrix(y_valuation, y_pred_list).ravel()

precision = tp/(tp + fp)
recall = tp/(tp + fn)
f1 = (2*precision*recall)/(precision+recall)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)


