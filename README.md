# 🎮 Enhanced Hand Gesture YouTube Controller


<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Control YouTube videos with hand gestures in real-time using computer vision and AI**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

A sophisticated computer vision application that enables touchless control of YouTube videos through intuitive hand gestures. Built with MediaPipe's hand tracking technology and powered by OpenCV, this system delivers real-time gesture recognition with sub-second latency.

### 🎯 Key Highlights

- **Real-time Processing**: Gesture detection at 30+ FPS with optimized CV pipeline
- **9 Gesture Commands**: Comprehensive control suite from play/pause to fullscreen
- **Robust Recognition**: Advanced finger state detection with stability filtering
- **Smart Cooldown**: Prevents accidental triggers with gesture stability tracking
- **Intuitive UI**: Live feedback overlay with help menu and FPS monitoring
- **Production-Ready**: Clean OOP architecture with error handling and logging

---

## ✨ Features

### 🖐️ Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| ✊ **Closed Fist** | Play/Pause | Toggle video playback |
| 👍 **Thumbs Up** | Volume Up | Increase volume by 10% |
| 👎 **Thumbs Down** | Volume Down | Decrease volume by 10% |
| ☝️ **Index Finger** | Next Video | Skip to next video in playlist |
| 🤙 **Pinky Finger** | Previous Video | Go back to previous video |
| ✌️ **Peace Sign** | Skip Forward | Jump ahead 10 seconds |
| 🤟 **Three Fingers** | Skip Backward | Rewind 10 seconds |
| ✋ **Open Palm** | Fullscreen | Toggle fullscreen mode |
| 👌 **OK Sign** | Mute/Unmute | Toggle audio |

### 🎨 Interactive UI

- **Live Gesture Display**: Real-time visualization of detected hand landmarks
- **Action Feedback**: Instant confirmation of executed commands
- **Help Menu**: Toggle-able overlay with gesture reference (Press 'H')
- **Performance Metrics**: Live FPS counter for monitoring system performance
- **Status Bar**: Keyboard shortcuts and system information

### 🔧 Technical Features

- **MediaPipe Integration**: Industry-standard hand tracking with 21-point landmark detection
- **Selenium Automation**: Reliable browser control with JavaScript injection
- **Gesture Stability**: Multi-frame validation to prevent false positives
- **Adaptive Detection**: Handedness-aware finger state analysis
- **Configurable Parameters**: Easy tuning of confidence thresholds and timings

---

## 🎬 Demo

### Visual Showcase

```
┌─────────────────────────────────────────┐
│  👋 Hand Gesture YouTube Controller     │
├─────────────────────────────────────────┤
│                                         │
│     [Live Camera Feed with Hand         │
│      Landmark Overlay]                  │
│                                         │
│  Current Gesture: Peace Sign            │
│  Action: ⏩ Skip +10s                   │
│                                         │
│  FPS: 32.5 | Press 'H' for Help        │
└─────────────────────────────────────────┘
```

### Sample Output

```bash
18:45:12 | INFO     | ✓ WebDriver initialized successfully
18:45:14 | INFO     | ✓ YouTube video loaded
18:45:15 | INFO     | ✓ Camera initialized
18:45:16 | INFO     | 🚀 Hand Gesture YouTube Controller started
18:45:18 | INFO     | Action: ▶ Playing
18:45:22 | INFO     | Action: Volume: 70%
18:45:25 | INFO     | Action: ⏩ +10s
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **Webcam** (USB or built-in)
- **Chrome Browser** (latest version)
- **ChromeDriver** ([Download here](https://chromedriver.chromium.org/downloads))

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MiyuruDilakshan/Enhanced-Hand-Gesture-YouTube-Controller.git
   cd hand-gesture-youtube-controller
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download ChromeDriver**
   - Visit [ChromeDriver Downloads](https://chromedriver.chromium.org/downloads)
   - Match your Chrome browser version
   - Extract and note the path

5. **Configure Path**
   ```python
   # Update in main() function
   CHROMEDRIVER_PATH = r"C:\path\to\chromedriver.exe"
   ```

### Requirements

```txt
opencv-python==4.8.1.78
mediapipe==0.10.7
selenium==4.15.2
numpy==1.24.3
```

---

## 💻 Usage

### Quick Start

```bash
python gesture_controller.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `H` | Toggle help menu |
| `R` | Reset gesture detection |
| `Q` | Quit application |

### Configuration

Customize behavior by modifying the `Config` class:

```python
@dataclass
class Config:
    CAMERA_WIDTH: int = 640           # Camera resolution
    DISPLAY_WIDTH: int = 800          # Display resolution
    GESTURE_COOLDOWN: float = 1.5     # Seconds between actions
    GESTURE_STABILITY_FRAMES: int = 3 # Frames for confirmation
    MIN_DETECTION_CONFIDENCE: float = 0.7  # Hand detection threshold
```

### Custom YouTube Video

```python
controller = GestureController(CHROMEDRIVER_PATH)
controller.youtube.open_youtube("https://www.youtube.com/watch?v=YOUR_VIDEO_ID")
controller.run()
```

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                   Main Application                       │
│                  GestureController                       │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼──────────┐
        │  Camera Input  │    │ YouTube Control  │
        │   cv2.VideoCapture   │   WebDriver     │
        └───────┬────────┘    └──────┬──────────┘
                │                     │
        ┌───────▼────────────┐   ┌───▼───────────┐
        │ Gesture Recognition│   │   Selenium    │
        │    MediaPipe       │   │  JavaScript   │
        └───────┬────────────┘   └───────────────┘
                │
        ┌───────▼────────┐
        │ Overlay Render │
        │     OpenCV     │
        └────────────────┘
```

### Core Components

#### 1. **GestureRecognizer**
- Hand landmark detection using MediaPipe
- Finger state analysis (extended vs. closed)
- Pattern matching for gesture classification
- Handedness-aware recognition

#### 2. **YouTubeController**
- Browser automation with Selenium WebDriver
- JavaScript injection for video control
- Safe script execution with error handling
- Comprehensive media controls

#### 3. **OverlayRenderer**
- Real-time UI overlay system
- Action feedback display
- Help menu rendering
- FPS and status monitoring

#### 4. **GestureController**
- Main application orchestration
- Gesture stability tracking
- Cooldown management
- Performance optimization

### Code Quality

- **Object-Oriented Design**: Clean separation of concerns
- **Type Hints**: Full type annotations for clarity
- **Error Handling**: Comprehensive exception management
- **Logging**: Structured logging for debugging
- **Documentation**: Docstrings for all classes and methods
- **Configuration**: Centralized settings with dataclass

---

## 🔬 Technical Deep Dive

### Gesture Recognition Algorithm

```python
1. Capture video frame from webcam
2. Convert to RGB and process with MediaPipe
3. Extract 21 hand landmarks (3D coordinates)
4. Analyze finger states:
   - Thumb: Horizontal extension check
   - Fingers: Vertical extension with pip-tip comparison
5. Pattern matching against gesture database
6. Apply stability filter (3-frame confirmation)
7. Execute action if cooldown elapsed
```

### Performance Optimizations

- **Frame Processing**: Separate capture and display resolutions
- **Gesture Stability**: Multi-frame validation reduces false positives
- **Action Cooldown**: Prevents rapid-fire triggering
- **Efficient Drawing**: Minimal overlay rendering
- **Buffer Management**: Single-frame buffer for minimal latency

---

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected**
```bash
# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**ChromeDriver version mismatch**
```bash
# Check Chrome version
chrome://version/

# Download matching ChromeDriver
# https://chromedriver.chromium.org/downloads
```

**Gesture not recognized**
- Ensure good lighting conditions
- Keep hand within camera frame
- Make clear, distinct gestures
- Check help menu for correct hand positions

**Low FPS performance**
- Close other camera applications
- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT` in Config
- Lower `MIN_DETECTION_CONFIDENCE` threshold

---

## 🚦 Roadmap

- [ ] **Multi-hand support**: Control with both hands simultaneously
- [ ] **Custom gesture training**: User-defined gesture creation
- [ ] **Mobile app version**: Android/iOS implementation
- [ ] **Streaming platform support**: Netflix, Spotify integration
- [ ] **Voice command integration**: Hybrid control system
- [ ] **Machine learning improvements**: Deep learning gesture models
- [ ] **Performance analytics**: Detailed gesture accuracy metrics

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints for new functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

Need help? Contact the developer:

### 👨‍💻 Developer Information

**Miyuru Dilakshan**

- 📧 **Email**: [Miyurudilakshan@gmail.com](mailto:Miyurudilakshan@gmail.com)
- 💬 **WhatsApp**: [+94 78 7517274](https://wa.me/94787517274)
- 🌐 **Website**: [miyuru.dev](https://miyuru.dev)
- 💼 **LinkedIn**: [linkedin.com/in/miyurudilakshan](https://www.linkedin.com/in/miyurudilakshan/)
- 🐙 **GitHub**: [github.com/miyurudilakshan](https://github.com/miyurudilakshan)
---

## 🙏 Acknowledgments

- **MediaPipe** - Google's hand tracking solution
- **OpenCV** - Computer vision library
- **Selenium** - Browser automation framework
- **Python Community** - For excellent documentation and support

---

## 📊 Project Stats

<div align="center">

![Language Count](https://img.shields.io/github/languages/count/yourusername/hand-gesture-youtube-controller)
![Code Size](https://img.shields.io/github/languages/code-size/yourusername/hand-gesture-youtube-controller)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/hand-gesture-youtube-controller)

</div>

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ and Python

</div>
