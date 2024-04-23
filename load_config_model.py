import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

actions = np.array(['golpe_derecha', 'golpe_izquierda','caminando_izquierda','caminando_derecha'])

def generate_model(weigths_path):
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(10,36)))
    lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(actions.shape[0], activation='softmax'))
    lstm_model.load_weights(weigths_path);

    return lstm_model;


def get_models_conf():
    models_conf = [
        {
            'model': generate_model('./latest.h5'),
            'sequence': [],
            'predictions':[]
        }
        ,
        {
            'model': generate_model('./latest.h5'),
            'sequence': [],
            'predictions':[]
        }
    ]
    return models_conf;