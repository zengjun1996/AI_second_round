from keras.models import load_model
import pandas as pd

y1 = pd.read_csv('data/Y_1.csv')
y2 = pd.read_csv('data/Y_2.csv')

model = load_model('model.h5')
x1 = model.predict(y1)
x2 = model.predict(y2)
x1.tofile('X_pre_1.bin')
x2.tofile('X_pre_2.bin')
