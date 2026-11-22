import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
EPOCHS = 100
BATCH_SIZE = 32


def load_data():
    print("[*] Loading data...")
    X = np.load(os.path.join("X.npy"))
    y = np.load(os.path.join("y.npy"))
    print(f"[*] Data X shape: {X.shape}")  # (num_samples, seq_len, num_features)
    print(f"[*] Number of samples: {X.shape[0]}")
    print(f"[*] Number of classes: {len(np.unique(y))}")
    return X, y


# Multi-class ROC curves
def plot_multiclass_roc(y_test, y_score, n_classes, class_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label='ROC curve of {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Random baseline diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig("roc_curve.png")
    print("[*] Saved roc_curve.png")
    plt.close()


# Training history plots
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig("training_history.png")
    print("[*] Saved training_history.png")
    plt.close()


def main():
    # Load dataset
    X, y = load_data()

    # Normalize features
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    X = X_scaled.reshape(num_samples, seq_len, num_features)

    # Encode labels
    encoder = LabelBinarizer()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    print(f"[*] Classes: {encoder.classes_}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=36, stratify=y
    )

    # Build 1D-CNN + LSTM hybrid model
    print("[*] Building model...")
    model = Sequential()

    # CNN Layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_len, num_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # LSTM Layer
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))

    # Output Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Training callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
    )
    plot_history(history)

    # Evaluation
    print("\n[*] Evaluating on Test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix (DL Model)")
    plt.savefig("confusion_matrix_dl.png")
    print("[*] Saved confusion_matrix_dl.png")
    plot_multiclass_roc(y_test, y_pred_prob, num_classes, encoder.classes_)

    # Save model and preprocessors
    model.save("traffic_model_dl.keras")
    joblib.dump(scaler, "scaler_dl.joblib")
    joblib.dump(encoder, "encoder_dl.joblib")
    print("[*] Saved: traffic_model_dl.keras, scaler_dl.joblib, encoder_dl.joblib")


if __name__ == "__main__":
    main()
