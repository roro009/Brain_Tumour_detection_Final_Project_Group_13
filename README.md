# **Brain Tumor Detection Using Deep Learning**

## **Project Overview**

This project aims to classify MRI brain images into four categories using Convolutional Neural Networks (CNNs):
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

By leveraging deep learning and data augmentation techniques, the project achieves robust tumor detection to aid early diagnosis and medical decision-making.

---

## **Dataset**

The dataset used in this project was sourced from **Kaggle** and includes MRI scans of human brains. The images are organized into four classes.

### **Dataset Structure**

| **Category**      | **Number of Images** |
|--------------------|----------------------|
| Glioma            | ~3,000               |
| Meningioma        | ~1,500               |
| Pituitary Tumor   | ~1,500               |
| No Tumor          | ~1,000               |
| **Total**         | ~7,023               |

- **Training Folder**: Images used for training the CNN model.  
- **Testing Folder**: Images used for evaluation and performance testing.

### **Image Details**
- **Format**: JPG/PNG  
- **Resolution**: 224x224 pixels (after preprocessing)  
- **Grayscale Images**: Standardized for consistency.  

---

## **Preprocessing Steps**

1. **Cropping**: Removal of unnecessary margins and noise.
2. **Resizing**: All images resized to **224x224 pixels**.
3. **Normalization**: Scaling pixel values to the range [0, 1].
4. **Augmentation**:  
   - Horizontal/Vertical flipping  
   - Rotation (up to 15 degrees)  
   - Zooming and translations (shifts)

---

## **Methodology**

The workflow follows these key steps:

1. **Data Preprocessing**:
   - Resizing and augmentation of images.
2. **Model Architecture**:
   - A Convolutional Neural Network (CNN) with 3 convolutional layers, max-pooling, dense layers, and a softmax classifier.
3. **Training**:
   - The model was trained using the Adam optimizer and categorical cross-entropy loss function.
4. **Evaluation**:
   - Performance metrics including accuracy, precision, recall, and F1-score were calculated.
5. **Visualization**:
   - Loss and accuracy curves, confusion matrix, and sample predictions.

---

## **Model Architecture**

The CNN model comprises:

- **Input Layer**: (224, 224, 1) grayscale input images  
- **Convolutional Layers**:  
   - Layer 1: 32 filters  
   - Layer 2: 64 filters  
   - Layer 3: 128 filters  
- **Pooling Layers**: Max-pooling with a 2x2 window.  
- **Flatten Layer**  
- **Dense Layers**:  
   - Fully connected layer with 256 neurons (ReLU activation).  
   - Dropout layer (50%).  
- **Output Layer**: Softmax activation with 4 neurons for class prediction.  

---

## **Results**

### **Performance Metrics**

| **Metric**        | **Glioma** | **Meningioma** | **Pituitary** | **No Tumor** | **Overall** |
|--------------------|-----------|---------------|--------------|-------------|-------------|
| **Precision**      | 0.64      | 0.96          | 0.72         | 0.94        | -           |
| **Recall**         | 0.95      | 0.32          | 0.92         | 0.89        | -           |
| **F1-Score**       | 0.77      | 0.48          | 0.81         | 0.92        | 0.78        |
| **Accuracy**       | -         | -             | -            | -           | **78%**     |

### **Visualization**

1. **Loss Curves**:
   - Training vs. Validation Loss (to assess overfitting).
2. **Accuracy Curves**:
   - Training vs. Validation Accuracy.
3. **Confusion Matrix**:
   - Visualizes model performance across all classes.
4. **Sample Predictions**:
   - Correctly classified and misclassified images.

