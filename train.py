# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Flatten, Dense, RepeatVector, Bidirectional, LSTM, TimeDistributed
)
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.mixed_precision import set_global_policy
from google.colab import drive
from PIL import Image

# Enable mixed precision for speed
set_global_policy('mixed_float16')

# ---- Step 1: Mount and verify Google Drive ----
def mount_drive():
    drive.mount('/content/drive', force_remount=True)
    root_dir = '/content/drive/MyDrive/Classroom/New Data/data'
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
    return root_dir

train_path = mount_drive()

# ---- Step 2: Parameters ----
n_chars = 'abcdefghijklmnopqrstuvwxyz '
char_to_idx = {c: i+1 for i, c in enumerate(n_chars)}  # 1-based
char_to_idx['<PAD>'] = 0
num_classes = len(char_to_idx)
seq_length = 10
input_shape = (32, 32, 1)
batch_size = 32

# ---- Step 3: Load and encode data ----
def load_data(root_dir):
    image_paths, labels = [], []
    for label in os.listdir(root_dir):
        subdir = os.path.join(root_dir, label)
        if not os.path.isdir(subdir):
            continue
        seq = [char_to_idx.get(ch, 0) for ch in label[:seq_length]]
        seq += [0] * (seq_length - len(seq))
        for fname in os.listdir(subdir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(subdir, fname))
                labels.append(seq)
    return image_paths, np.array(labels, dtype=np.int32)

paths, labels = load_data(train_path)
if not paths:
    print(f"No images found in {train_path}. Generating a small temporary dataset...")
    temp_path = '/content/temp_dataset/training_words'
    os.makedirs(temp_path, exist_ok=True)
    for word in ['hello', 'world', 'test', 'data']:
        wd = os.path.join(temp_path, word)
        os.makedirs(wd, exist_ok=True)
        for i in range(3):
            img_arr = np.ones((32,32), dtype=np.uint8) * 255
            img_arr[i*8:(i+1)*8, :16] = 0
            Image.fromarray(img_arr).save(os.path.join(wd, f'{word}_{i}.png'))
    print(f"Using temporary dataset at {temp_path}")
    train_path = temp_path
    paths, labels = load_data(train_path)

# Split into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42
)

# ---- Step 4: Build tf.data pipelines ----
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [32, 32])
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths))
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

# ---- Step 5: Define the model ----
def build_model():
    inp = Input(shape=input_shape, name='image')
    x = inp
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = RepeatVector(seq_length)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    out = TimeDistributed(Dense(num_classes, activation='softmax', dtype='float32'))(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

# ---- Step 6: Callbacks ----
class PrintMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}")

callbacks = [
    ModelCheckpoint('best_seq2seq_ocr.h5', monitor='val_loss', save_best_only=True),
    PrintMetrics()
]

# ---- Step 7: Train ----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Ensure this is set to 50
    callbacks=callbacks,
    verbose=1  # Set verbose to 1 to see the training output
)

# ---- Step 8: Evaluation ----
preds_list, true_list = [], []
for imgs, lbls in val_ds:
    preds = model.predict(imgs, verbose=0)
    preds_list.append(np.argmax(preds, axis=-1).flatten())
    true_list.append(np.argmax(lbls.numpy(), axis=-1).flatten())
preds_arr = np.concatenate(preds_list)
true_arr = np.concatenate(true_list)
char_acc = np.mean(preds_arr == true_arr)
f1 = f1_score(true_arr, preds_arr, average='weighted')
print(f"Character Accuracy: {char_acc:.4f}, F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(true_arr, preds_arr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# ---- Step 9: Training Curves ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
