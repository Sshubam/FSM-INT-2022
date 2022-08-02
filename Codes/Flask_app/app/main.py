import math
from flask import *
import tensorflow as tf
import numpy as np

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

length_of_sequence = 2000
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv1D(32, 8, padding='same',input_shape=(2000, 1),activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(2, padding='same'))
model.add(tf.keras.layers.Conv1D(64, 8, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(2, padding='same'))
model.add(tf.keras.layers.Conv1D(128, 8, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(2, padding='same'))
model.add(tf.keras.layers.Conv1D(256, 8, padding='same', activation='relu'))
model.add(tf.keras.layers.LSTM(64, dropout = 0.2,recurrent_dropout = 0.5))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.load_weights('app/lstm_model_weights.best.hdf5')
@app.route('/predict', methods=['POST'])
def predict():
    csv = request.files['csv']
    csv_path = "/tmp/" + csv.filename
    csv.save(csv_path)
    from numpy import genfromtxt
    data = genfromtxt(csv_path, delimiter=',')
    sample = np.expand_dims(data, axis = 0)
    sample = np.expand_dims(sample, axis = 2)
    predicted=model.predict(sample).item()
    predicted=round(predicted, 3)
        
    return render_template('index.html', prediction = predicted)
    
if __name__ == "__main__":
    app.debug = True
    app.run()