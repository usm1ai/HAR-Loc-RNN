from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_rnn_activity_classification(X, y, input_shape, epochs=20, batch_size=32):
    y_categorical = to_categorical(y, num_classes=8)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(8, activation='softmax')  # 8 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def train_rnn_location_classification(X, y, input_shape, epochs=20, batch_size=32):
    y_categorical = to_categorical(y, num_classes=8)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    model = Sequential([
        SimpleRNN(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.4),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(8, activation='softmax')  # 8 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model
