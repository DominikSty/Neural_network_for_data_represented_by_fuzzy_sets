from keras.models import load_model
import numpy as np

# loading model from file
model = load_model('model.h5')

# preparation of new test data
x_new = np.random.rand(10, 2)  # compartment [0, 1]
# border points
# x_new = [[0, 0],[1, 1]]
print(x_new)

# prediction of values for new data
y_pred = model.predict(x_new)
print(y_pred)
