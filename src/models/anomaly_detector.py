import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.regularizers import l1

def build_autoencoder(input_dim):
    """Builds a Sparse Autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu', activity_regularizer=l1(0.0001))(input_layer)
    encoder = Dense(32, activation='relu', activity_regularizer=l1(0.0001))(encoder)
    encoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def build_lstm_autoencoder(seq_len, n_features):
    """Builds a Recurrent Autoencoder (LSTM-based) for time-series data."""
    inputs = Input(shape=(seq_len, n_features))
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu')(encoded)
    decoded = RepeatVector(seq_len)(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train_normal, X_test, epochs=50, batch_size=32):
    """Trains the autoencoder model on normal traffic only."""
    print("\n--- Training Sparse Autoencoder ---")
    history = model.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )
    return history

def save_model(model, path):
    """Saves the trained model to a specified path."""
    model.save(path)
    print(f"✅ Model saved to {path}")

def load_model(path):
    """Loads a pre-trained model from a specified path."""
    return tf.keras.models.load_model(path)

def save_model(model, path):
    """Saves the trained model to a specified path."""
    model.save(path)
    print(f"✅ Model saved to {path}")

def load_model(path):
    """Loads a pre-trained model from a specified path."""
    return tf.keras.models.load_model(path)