

# 🖼️ Image Classification App using MobileNetV2

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen.svg)](https://app-web-application-sjm9jaxnlg7bw4phzsfq69.streamlit.app/)

This is a web-based image classification app built with [Streamlit](https://streamlit.io/) and powered by the pre-trained MobileNetV2 model from [TensorFlow](https://www.tensorflow.org/). The app allows users to upload an image and classify it into top-3 categories with corresponding confidence scores.

### 🚀 Live Demo

Check out the live application [here](https://app-web-application-sjm9jaxnlg7bw4phzsfq69.streamlit.app/).

---

## 📸 Features

- **Image Upload**: Easily upload images in `jpg`, `jpeg`, or `png` formats.
- **Real-time Classification**: The app uses MobileNetV2 to classify the uploaded image and displays the top-3 predictions with corresponding confidence scores.
- **Progress Feedback**: Users can track the classification progress with a progress bar.
- **Graphical Representation**: A bar graph displays the prediction probabilities for the top-3 categories.

## 🧑‍💻 Technologies Used

- **Framework**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) with the MobileNetV2 pre-trained model
- **Image Processing**: [Pillow](https://python-pillow.org/) (PIL)
- **Visualization**: [Matplotlib](https://matplotlib.org/)

---

## 🛠️ Installation & Setup

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies:
Make sure you have Python 3.7+ installed. Then, run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the app locally:
```bash
streamlit run app.py
```

The app should now be running on your local machine, typically at `http://localhost:8501`.

---

## 🖼️ How It Works

1. **Upload an Image**: Go to the sidebar, upload an image in `jpg`, `jpeg`, or `png` format.
2. **Preprocessing**: The image is resized to `224x224` pixels and preprocessed to fit the MobileNetV2 model input format.
3. **Prediction**: The model generates the top-3 predictions, which are then displayed alongside a bar graph showing the prediction probabilities.
4. **View Results**: See the classified labels along with their corresponding confidence scores.




