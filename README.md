# 🖼️ Image Caption Generator using InceptionV3 and LSTM

This project implements an **Image Caption Generator** using deep learning. It extracts image features with **InceptionV3** and generates natural language captions via an **LSTM** network. The model is trained on the **Flickr8k** dataset and uses a **greedy search** for inference on new images.



---

## 📁 Project Structure

```
├── data/
│   ├── archive.zip                 # Flickr8k dataset (images + captions)
│   ├── Flickr_8k.trainImages.txt   # Training image IDs
│   ├── Flickr_8k.testImages.txt    # Test image IDs
│   ├── glove.6B.200d.txt           # Pre-trained GloVe word embeddings
│   └── model_checkpoint.h5         # Saved model weights (best checkpoint)
├── image_caption_generator.py      # Core script for training and inference
└── README.md                       # This documentation file
```

---

## ⚙️ Requirements

### Python Environment
- Python 3.8+
- TensorFlow 2.x
- Keras (included with TensorFlow)
- NumPy
- Pandas
- OpenCV (cv2)
- Matplotlib

### Installation
Run the following command to install dependencies:

```bash
pip install tensorflow numpy pandas opencv-python matplotlib
```

For GloVe embeddings, download `glove.6B.200d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and place it in the `data/` folder.

---

## 🚀 Quick Start

### 1. Setup Dataset
1. Download the **Flickr8k** dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Unzip `archive.zip` into the `data/` folder.
3. Update paths in `image_caption_generator.py` to match your local setup (e.g., dataset directory).

If using Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Run Training
Execute the script to train the model:
```bash
python image_caption_generator.py
```
- **Epochs**: 50 (adjustable)
- **Batch Size**: 256
- Saves the best model to `data/model_checkpoint.h5`

### 3. Generate Captions (Inference)
Load a trained model and generate captions for new images:
```python
# Example usage in the script
encoded_img = encode('/path/to/your/image.jpg', encoding_model)
caption = greedy_search(encoded_img)
print("Generated Caption:", caption)
```

Display the result:
```python
import cv2
from google.colab.patches import cv2_imshow  # For Colab; use plt.imshow() locally

image = cv2.imread('/path/to/your/image.jpg')
cv2_imshow(image)
print(f"Caption: {caption}")
```

---

## 🧠 Model Architecture & Workflow

### Key Components
1. **Image Feature Extraction**:
   - Uses pre-trained **InceptionV3** (without top layers).
   - Resizes images to 299x299 pixels.
   - Outputs 2048-dimensional feature vectors.

2. **Text Processing**:
   - Cleans captions: lowercase, remove punctuation, add `<startseq>` and `<endseq>` tokens.
   - Tokenizes vocabulary (~8k words from Flickr8k).
   - Uses **GloVe 200d** embeddings for word representations.

3. **Data Preparation**:
   - Split: 98% train, 2% test.
   - Pads sequences to max length (e.g., 35 tokens).
   - One-hot encodes targets for categorical loss.

4. **Neural Network**:
   - **Encoder**: InceptionV3 → Dense(256) → RepeatVector(max_len).
   - **Decoder**: Embedding → LSTM(256, return_sequences=True) → Dropout → TimeDistributed(Dense(vocab_size, activation='softmax')).
   - Merge: Add encoder output to LSTM input for attention-like fusion.

### Training Details
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: ModelCheckpoint (save best by val_loss)

Example compilation and training code:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('data/model_checkpoint.h5', monitor='val_loss', save_best_only=True)
model.fit([X1, X2], y, epochs=50, batch_size=256, validation_split=0.02, callbacks=[checkpoint])
```

### Inference: Greedy Search
- Starts with `<startseq>`.
- Predicts next word with argmax (greedy).
- Stops at `<endseq>` or max length.

---

## 📝 Important Notes
- **Custom Images**: Ensure inputs are RGB JPEG/PNG, resized to 299x299.
- **GloVe Loading**: Verify embedding matrix shape matches (vocab_size, 200).
- **Performance**: On CPU, training takes ~2-3 hours for 50 epochs. Use GPU for faster runs.
- **Limitations**: Captions may be repetitive; consider beam search for better diversity.
- **Troubleshooting**:
  - Shape errors? Check padding and embedding dims.
  - OOM? Reduce batch size to 128.

---

## 🔗 Resources & References
- **Dataset**: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **InceptionV3 Docs**: [Keras Applications](https://keras.io/api/applications/inceptionv3/)
- **GloVe Embeddings**: [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- **Full Tutorial Inspiration**: [Image Captioning with Keras](https://www.tensorflow.org/tutorials/text/image_captioning)

---

## 👥 Contributors
- **Abhiram G** - Lead Developer
- **Varun S** - Data Preprocessing
- **Md Faraz** - Model Training
- **Vivek Prabhu** - Documentation

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if it helps your project!** Questions? Open an issue or reach out.

---

