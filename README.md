# AI-Based Inventory Counting Automation

This project is an AI-powered inventory counting system that utilizes object detection to automate inventory tracking. It is built using Flask, YOLOv8, and OpenCV, with a front-end dashboard for visualizing detections.

## Features
- Object Detection: Uses YOLOv8 to detect objects in images.
- Image Enhancement: Improves image quality before detection.
- Dashboard Visualization: Displays object counts in a user-friendly UI.
- Flask API: Backend service for processing images and returning detected objects.
- Interactive UI: Upload images and view detection results in real-time.

## Tech Stack
- Backend: Flask, YOLOv8, OpenCV, NumPy
- Frontend: HTML, CSS, JavaScript, Axios
- Database: Included (Virtually through PostgreSQL)
## Installation

### Prerequisites
- Python 3.8+
- Python libraries
- pip

### Setup Instructions
1. Clone this repository:
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

2. Install dependencies:
pip install -r requirements.txt

3. Download YOLOv8 model:
mkdir models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..

4. Run the Flask server:
python app.py

5.Open index.html in a browser or use a local server:
python -m http.server 8000  # Runs a simple server

The backend provides an API for object detection:
Endpoint: POST /detect

Request Body:
{
  "image": "<base64-encoded image>"
}

Response Example:
{
  "processed_image": "<base64-encoded image>",
  "object_counts": {
    "bottle": 3,
    "box": 5
  }
}

Project Structure:
├── app.py               # Flask backend
├── index.html           # Main frontend UI
├── analysis.html        # Analysis page
├── .env                 # Environment variables (not committed)
├── static/              # CSS, JS files
└── README.md            # Project documentation
