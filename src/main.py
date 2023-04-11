from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# preparation of random data
x = np.random.rand(1000, 2)  # compartment [0, 1]
y = np.array([1 if xi[0]**2 + xi[1]**2 < 0.5 else 0 for xi in x])  # 1 or 0

print(x)
# input("Press enter to continue...")
print(y)
# input("Press enter to continue...")

# visualization of input data
class0 = plt.scatter(x[:, 0][y==0], x[:, 1][y==0], c='c', label="Class 0 (doesn't belong)")
class1 = plt.scatter(x[:, 0][y==1], x[:, 1][y==1], c='m', label='Class 1 (belongs)')
plt.legend(handles=[class0, class1], loc='upper left')
plt.title('Input data visualization \n [0,0] - center of the circle, 0.5 - circle radius')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

# division into a teaching and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# model definition
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model training
history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_test, y_test))

# save model to file
model.save('model.h5')

# display of accuracy and loss for learner and test data
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# display of accuracy and loss for learning and test data on graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Accuracy history')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['train', 'test'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Loss history')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train', 'test'], loc='upper left')

plt.show()
