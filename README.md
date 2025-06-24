Yoga Pose Detection Project (In Progress)

    ⚠️ This project is still under development. More features, improvements, and optimizations are being added.

📌 Description

This is a computer vision and machine learning project that uses MediaPipe and OpenCV to recognize different yoga poses via webcam. The system is trained on custom pose data and allows:

    Real-time prediction

    Teaching the user how to perform each pose

    A game mode where users try to match poses

    Adding new poses dynamically

🚀 Features

✅ Real-time Pose Detection
→ Detects body position via webcam using MediaPipe and predicts yoga poses using a trained Random Forest classifier.

✅ Teaching Mode
→ Displays an image and short description to help the user learn how to do each pose.

✅ Game Mode (Work in Progress)
→ A random pose is shown to the user, and they must try to perform it within a few seconds. The model gives feedback on how well the user matched the pose.

✅ Custom Pose Training
→ You can collect your own data for new poses using collect_data.py, then retrain the model.

✅ Unknown Pose Handling
→ If the model is unsure, it labels the pose as "Unknown" to avoid false predictions.
→ Evaluate the model's accuracy on each pose after training.



## 🧘 Example Pose: Pose tracking

This is how the Pose tracking looks:

![Pose](images/pose.jpg)

