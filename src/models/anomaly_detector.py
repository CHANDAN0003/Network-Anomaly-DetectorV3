import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed, Lambda
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

def build_autoencoder(input_dim):
    """Builds a Sparse Autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu', activity_regularizer=l1(0.0001), name="enc_dense_64")(input_layer)
    encoder = Dense(32, activation='relu', activity_regularizer=l1(0.0001), name="enc_dense_32")(encoder)
    encoder = Dense(16, activation='relu', name="bottleneck")(encoder)
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

def train_model(model, X_train_normal, X_test, epochs=100, batch_size=32):
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

def create_model(optimizer='adam', loss='mean_squared_error'):
    input_dim = 100  # Example input dimension, adjust as needed
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu', activity_regularizer=l1(0.0001))(input_layer)
    encoder = Dense(32, activation='relu', activity_regularizer=l1(0.0001))(encoder)
    encoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder

def tune_hyperparameters(X_train, param_grid):
    """Tunes hyperparameters using GridSearchCV."""
    model = KerasRegressor(build_fn=create_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train, X_train)
    print("Best parameters found: ", grid_result.best_params_)
    print("Best score: ", grid_result.best_score_)
    return grid_result.best_estimator_


def get_encoder_from_autoencoder(autoencoder: Model) -> Model:
    """Return an encoder model that outputs the bottleneck layer.

    Assumes the autoencoder built by build_autoencoder with layer named 'bottleneck'.
    If the name is not found, falls back to layer index 3 (Input, 64, 32, 16).
    """
    try:
        bottleneck_layer = autoencoder.get_layer("bottleneck")
        return Model(inputs=autoencoder.input, outputs=bottleneck_layer.output)
    except Exception:
        # Fallback to index 3 if custom model used with same topology
        if len(autoencoder.layers) > 3:
            return Model(inputs=autoencoder.input, outputs=autoencoder.layers[3].output)
        else:
            raise ValueError("Cannot locate bottleneck layer to build encoder.")


def build_vae(input_dim: int, latent_dim: int = 16):
    """Build a simple Variational Autoencoder (optional, not used by default)."""
    from tensorflow.keras.layers import Layer
    from tensorflow.keras import backend as K

    inputs = Input(shape=(input_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(32, activation='relu')(h)
    z_mean = Dense(latent_dim, name="z_mean")(h)
    z_log_var = Dense(latent_dim, name="z_log_var")(h)

    def sampling(args):
        z_mean_, z_log_var_ = args
        epsilon = K.random_normal(shape=(K.shape(z_mean_)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean_ + K.exp(0.5 * z_log_var_) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    h_dec = Dense(32, activation='relu')(z)
    h_dec = Dense(64, activation='relu')(h_dec)
    outputs = Dense(input_dim, activation='sigmoid')(h_dec)

    vae = Model(inputs, outputs)

    # VAE loss
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer='adam')
    return vae
