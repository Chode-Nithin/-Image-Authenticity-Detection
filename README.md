# Image and Video Authenticity Detection

This project focuses on developing tools for image authenticity detection and face extraction from videos. In today's digital age, the proliferation of fake images and videos presents significant challenges in various domains such as media forensics, content moderation, and security. This project aims to address these challenges by leveraging deep learning techniques for image classification and video analysis.

## Key Features

### Image Authenticity Detection

- Implemented a Convolutional Neural Network (CNN) model using TensorFlow/Keras to classify images as real or fake.
- Transfer learning techniques were employed by fine-tuning a pre-trained model for improved performance.
- **Streamlit Interface**: Created a user-friendly interface using Streamlit, allowing users to upload images for authenticity verification without requiring any coding knowledge.

### Video Analysis and Face Extraction

- Utilized computer vision techniques to extract frames from videos.
- Filtered out videos with insufficient data.
- Detected faces within each frame using the `face_recognition` library.
- Extracted faces were resized and written to new video files for isolated face analysis.

## Applications

- **Media Forensics**: Detection of fake images and videos for combating misinformation and ensuring content authenticity.
- **Content Moderation**: Enhancing security measures by verifying the authenticity of uploaded images and videos on online platforms.
- **Video Analytics**: Facilitating tasks such as face recognition, video summarization, and surveillance through face extraction from real videos.

## Future Work

- Integration of advanced face recognition algorithms for more robust face analysis.
- Enhancement of the image authenticity detection model with larger datasets and advanced deep learning architectures.
- Deployment of the project as a web application for wider accessibility and usability.

## Contributors

- Chode Nithin/ @Chode-Nithin
- Chokkapu Monisha/ @chokkapumonisha

## Contributions

Contributions to this project are welcome! Whether it's bug fixes, feature enhancements, or new ideas, feel free to contribute by submitting pull requests.

## Feedback and Support

For any issues or suggestions, please contact nithinchode@gmail.com. We appreciate your feedback!

---
