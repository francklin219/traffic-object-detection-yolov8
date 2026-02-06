\# 🚦 Traffic Object Detection with YOLOv8



This project implements a \*\*traffic object detection system\*\* using \*\*YOLOv8\*\*, trained on a subset of the \*\*BDD100K dataset\*\*.  

The model is capable of detecting multiple road-related objects such as vehicles, pedestrians, riders, traffic lights, and traffic signs, and was trained end-to-end on Google Colab with GPU acceleration.



---



\## 📖 Project Overview



Road traffic analysis is a key component in intelligent transportation systems and autonomous driving.  

The goal of this project is to:



\- Train a \*\*YOLOv8 object detection model\*\*

\- Convert the \*\*BDD100K annotations\*\* into YOLO format

\- Evaluate model performance across multiple classes

\- Demonstrate inference on real-world traffic videos

\- Provide reproducible training results and visual analytics



---



\## 🧠 Model \& Dataset



\### Model

\- \*\*Architecture:\*\* YOLOv8n (Ultralytics)

\- \*\*Framework:\*\* PyTorch

\- \*\*Input size:\*\* 640 × 640

\- \*\*Epochs:\*\* 20

\- \*\*Optimizer:\*\* SGD (auto-selected by YOLOv8)

\- \*\*Hardware:\*\* NVIDIA Tesla T4 (Google Colab)



\### Dataset

\- \*\*Dataset:\*\* BDD100K

\- \*\*Task:\*\* Object Detection

\- \*\*Annotations:\*\* Bounding boxes

\- \*\*Classes (10):\*\*

&nbsp; - person

&nbsp; - rider

&nbsp; - car

&nbsp; - truck

&nbsp; - bus

&nbsp; - train

&nbsp; - motorcycle

&nbsp; - bicycle

&nbsp; - traffic light

&nbsp; - traffic sign



---



\## 📂 Project Structure



traffic-object-detection-yolov8/

├── src/

│ ├── train.py # Training script

│ ├── predict.py # Inference on images/videos

│ └── convert\_bdd100k\_to\_yolo.py# Dataset conversion script

│

├── assets/

│ ├── traffic\_yolo\_demo.mp4 # Inference demo video

│ ├── map\_curves.png # mAP curves

│ ├── loss\_curves.png # Training loss curves

│ └── confusion\_matrix.png # Confusion matrix

│

├── runs/ # YOLO training outputs (partial)

│

├── README.md

├── .gitignore





> ⚠️ \*\*Note:\*\*  

> The raw dataset (`BDD100K`) and intermediate training files are intentionally excluded from version control.



---



\## 📊 Training Results



\### 🔹 Mean Average Precision

\- \*\*mAP@50:\*\* ≈ \*\*0.46\*\*

\- \*\*mAP@50–95:\*\* ≈ \*\*0.26\*\*



These results indicate a solid baseline performance given the model size (YOLOv8n) and limited training epochs.



\#### mAP Curves

!\[mAP curves](assets/map\_curves.png)



---



\### 🔹 Loss Curves

Training and validation losses decrease consistently, showing stable convergence without overfitting.



!\[Loss curves](assets/loss\_curves.png)



---



\### 🔹 Confusion Matrix

The confusion matrix highlights strong performance on dominant classes such as \*\*cars\*\*, \*\*traffic signs\*\*, and \*\*traffic lights\*\*, while rarer classes (e.g. \*train\*) remain more challenging.



!\[Confusion matrix](assets/confusion\_matrix.png)



---



\## 🎥 Inference Demo



The trained model was tested on a real-world traffic video.



▶️ \*\*Demo video:\*\*  

`assets/traffic\_yolo\_demo.mp4`



The demo shows:

\- Real-time bounding box predictions

\- Multi-class detection

\- Robust performance in dense urban traffic scenes



---



\## ⚙️ How to Run



\### 1️⃣ Install dependencies

```bash

pip install ultralytics opencv-python matplotlib pandas

2️⃣ Train the model

python src/train.py

3️⃣ Run inference on a video

python src/predict.py --source path/to/video.mp4 --weights best.pt

🚀 Key Takeaways

Successfully trained a YOLOv8 model for multi-class traffic detection



Built a full ML pipeline: data conversion → training → evaluation → inference



Demonstrated solid results with limited compute resources



Project structured for reproducibility and professional presentation



🔮 Future Improvements

Train for more epochs and/or use a larger model (YOLOv8m/l)



Apply data augmentation strategies



Balance rare classes (e.g. train, rider)



Export the model to ONNX / TensorRT for deployment



Real-time inference benchmarking (FPS evaluation)



👤 Author

Franck

Machine Learning \& Computer Vision Project

YOLOv8 · Object Detection · Traffic Analysis

