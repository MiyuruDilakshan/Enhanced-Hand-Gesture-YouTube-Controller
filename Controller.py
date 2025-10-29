"""
Enhanced Hand Gesture YouTube Controller
========================================
Control YouTube videos with hand gestures using your webcam — 
no mouse, no keyboard. Real-time AI with MediaPipe + OpenCV + Selenium. 
Fist to pause, thumbs up for volume, peace sign to skip. Built for speed, precision, and wow-factor.

Author: Miyuru Dilakshan
Version: 1.0.0
License: MIT
"""

import cv2
import mediapipe as mp
import time
import math
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import numpy as np


# Configuration
@dataclass
class Config:
    """Application configuration parameters"""
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    DISPLAY_WIDTH: int = 800
    DISPLAY_HEIGHT: int = 600
    FPS: int = 30
    GESTURE_COOLDOWN: float = 1.5
    GESTURE_STABILITY_FRAMES: int = 3
    OVERLAY_TIMEOUT: float = 2.0
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5


class Gesture(Enum):
    """Enumeration of supported gestures"""
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    INDEX = "index"
    PINKY = "pinky"
    PEACE = "peace"
    THREE = "three"
    OPEN_PALM = "open_palm"
    OK = "ok"
    UNKNOWN = "unknown"
    NO_HAND = "no_hand"


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class YouTubeController:
    """
    Handles YouTube video control via Selenium WebDriver.
    Provides methods for play/pause, volume control, navigation, and fullscreen.
    """
    
    def __init__(self, chromedriver_path: str):
        self.driver: Optional[webdriver.Chrome] = None
        self.chromedriver_path = chromedriver_path
        self._initialize_driver()
    
    def _initialize_driver(self) -> None:
        """Initialize Chrome WebDriver with optimized settings"""
        try:
            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--mute-audio")
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
            options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(self.chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.implicitly_wait(10)
            logger.info("✓ WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize WebDriver: {e}")
            raise
    
    def open_youtube(self, video_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ") -> bool:
        """Open YouTube video and wait for page load"""
        try:
            self.driver.get(video_url)
            time.sleep(3)
            logger.info(f"✓ YouTube video loaded")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to open YouTube: {e}")
            return False
    
    def _execute_script(self, script: str) -> Optional[str]:
        """Execute JavaScript with comprehensive error handling"""
        try:
            return self.driver.execute_script(script)
        except WebDriverException as e:
            logger.debug(f"Script execution warning: {e}")
            return None
    
    def play_pause(self) -> Optional[str]:
        """Toggle video play/pause state"""
        return self._execute_script("""
            const video = document.querySelector("video");
            if (video) {
                if (video.paused) {
                    video.play();
                    return "▶ Playing";
                } else {
                    video.pause();
                    return "⏸ Paused";
                }
            }
            return null;
        """)
    
    def volume_up(self) -> Optional[str]:
        """Increase volume by 10%"""
        return self._execute_script("""
            const video = document.querySelector("video");
            if (video) {
                video.volume = Math.min(1, video.volume + 0.1);
                return `Volume: ${Math.round(video.volume * 100)}%`;
            }
            return null;
        """)
    
    def volume_down(self) -> Optional[str]:
        """Decrease volume by 10%"""
        return self._execute_script("""
            const video = document.querySelector("video");
            if (video) {
                video.volume = Math.max(0, video.volume - 0.1);
                return `Volume: ${Math.round(video.volume * 100)}%`;
            }
            return null;
        """)
    
    def mute_unmute(self) -> Optional[str]:
        """Toggle audio mute state"""
        return self._execute_script("""
            const video = document.querySelector("video");
            if (video) {
                video.muted = !video.muted;
                return video.muted ? "🔇 Muted" : "🔊 Unmuted";
            }
            return null;
        """)
    
    def skip_forward(self, seconds: int = 10) -> Optional[str]:
        """Skip forward by specified seconds"""
        return self._execute_script(f"""
            const video = document.querySelector("video");
            if (video) {{
                video.currentTime = Math.min(video.duration, video.currentTime + {seconds});
                return "⏩ +{seconds}s";
            }}
            return null;
        """)
    
    def skip_backward(self, seconds: int = 10) -> Optional[str]:
        """Skip backward by specified seconds"""
        return self._execute_script(f"""
            const video = document.querySelector("video");
            if (video) {{
                video.currentTime = Math.max(0, video.currentTime - {seconds});
                return "⏪ -{seconds}s";
            }}
            return null;
        """)
    
    def next_video(self) -> Optional[str]:
        """Navigate to next video in playlist"""
        return self._execute_script("""
            const nextBtn = document.querySelector('.ytp-next-button');
            if (nextBtn && !nextBtn.disabled) {
                nextBtn.click();
                return "⏭ Next video";
            }
            return "Next unavailable";
        """)
    
    def prev_video(self) -> Optional[str]:
        """Navigate to previous video in playlist"""
        return self._execute_script("""
            const prevBtn = document.querySelector('.ytp-prev-button');
            if (prevBtn && !prevBtn.disabled) {
                prevBtn.click();
                return "⏮ Previous video";
            }
            return "Previous unavailable";
        """)
    
    def toggle_fullscreen(self) -> Optional[str]:
        """Toggle fullscreen mode"""
        return self._execute_script("""
            const fullscreenBtn = document.querySelector('.ytp-fullscreen-button');
            if (fullscreenBtn) {
                fullscreenBtn.click();
                return "⛶ Fullscreen toggled";
            }
            return null;
        """)
    
    def close(self) -> None:
        """Safely close the browser"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("✓ Browser closed")
            except:
                pass


class GestureRecognizer:
    """
    Advanced hand gesture recognition using MediaPipe Hands.
    Implements robust finger state detection and gesture classification.
    """
    
    def __init__(self, config: Config):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
    
    def _get_finger_states(self, landmarks) -> List[int]:
        """
        Analyze finger extension states with improved accuracy.
        Returns [thumb, index, middle, ring, pinky] where 1=extended, 0=closed
        """
        fingers = []
        
        # Determine hand orientation
        wrist_x = landmarks[0].x
        palm_center_x = np.mean([landmarks[i].x for i in [5, 9, 13, 17]])
        is_right_hand = palm_center_x < wrist_x
        
        # Thumb detection (horizontal extension)
        thumb_extended = (landmarks[4].x < landmarks[2].x) if is_right_hand else (landmarks[4].x > landmarks[2].x)
        fingers.append(1 if thumb_extended else 0)
        
        # Four fingers (vertical extension)
        tip_indices = [8, 12, 16, 20]
        pip_indices = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(tip_indices, pip_indices):
            tip_y = landmarks[tip_idx].y
            pip_y = landmarks[pip_idx].y
            is_extended = tip_y < pip_y and abs(tip_y - pip_y) > 0.025
            fingers.append(1 if is_extended else 0)
        
        return fingers
    
    def _calculate_distance(self, p1, p2) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _detect_ok_gesture(self, landmarks) -> bool:
        """Detect OK sign (thumb-index circle) with high precision"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        thumb_index_dist = self._calculate_distance(thumb_tip, index_tip)
        index_middle_dist = self._calculate_distance(index_tip, middle_tip)
        
        fingers = self._get_finger_states(landmarks)
        other_fingers_extended = sum(fingers[2:]) >= 2
        
        return (thumb_index_dist < 0.05 and 
                thumb_index_dist < index_middle_dist * 0.5 and 
                other_fingers_extended)
    
    def _detect_thumbs_down(self, landmarks, fingers: List[int]) -> bool:
        """Detect thumbs down gesture"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        thumb_pointing_down = thumb_tip.y > thumb_mcp.y and thumb_tip.y > wrist.y
        other_fingers_closed = sum(fingers[1:]) == 0
        
        return thumb_pointing_down and other_fingers_closed
    
    def recognize_gesture(self, landmarks) -> Gesture:
        """
        Main gesture recognition pipeline with pattern matching.
        Returns the detected gesture enum value.
        """
        if not landmarks:
            return Gesture.NO_HAND
        
        fingers = self._get_finger_states(landmarks)
        
        # Priority-based gesture detection
        if self._detect_ok_gesture(landmarks):
            return Gesture.OK
        
        if self._detect_thumbs_down(landmarks, fingers):
            return Gesture.THUMBS_DOWN
        
        # Pattern matching
        gesture_patterns = {
            Gesture.FIST: [0, 0, 0, 0, 0],
            Gesture.THUMBS_UP: [1, 0, 0, 0, 0],
            Gesture.INDEX: [0, 1, 0, 0, 0],
            Gesture.PINKY: [0, 0, 0, 0, 1],
            Gesture.PEACE: [0, 1, 1, 0, 0],
            Gesture.THREE: [0, 1, 1, 1, 0],
            Gesture.OPEN_PALM: [1, 1, 1, 1, 1],
        }
        
        # Exact match
        for gesture, pattern in gesture_patterns.items():
            if fingers == pattern:
                return gesture
        
        # Fuzzy matching for robustness
        if sum(fingers) >= 4:
            return Gesture.OPEN_PALM
        elif fingers == [0, 1, 1, 0, 0] or (fingers[1] and fingers[2] and sum(fingers) == 2):
            return Gesture.PEACE
        elif fingers[1:4] == [1, 1, 1] and sum(fingers) == 3:
            return Gesture.THREE
        
        return Gesture.UNKNOWN
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> None:
        """Draw hand skeleton with custom styling"""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style()
        )


class OverlayRenderer:
    """Handles all UI overlay rendering on video frames"""
    
    def __init__(self, config: Config):
        self.config = config
        self.overlay_text = ""
        self.overlay_time = 0.0
        self.show_help = False
    
    def set_message(self, text: str) -> None:
        """Display a timed message overlay"""
        self.overlay_text = text
        self.overlay_time = time.time()
        logger.info(f"Action: {text}")
    
    def toggle_help(self) -> None:
        """Toggle help menu visibility"""
        self.show_help = not self.show_help
    
    def render(self, frame: np.ndarray, gesture: Gesture, fps: float) -> None:
        """Render all overlay elements on the frame"""
        self._render_status_bar(frame, fps)
        self._render_gesture_label(frame, gesture)
        
        if time.time() - self.overlay_time < self.config.OVERLAY_TIMEOUT:
            self._render_action_message(frame)
        
        if self.show_help:
            self._render_help_menu(frame)
    
    def _render_text_box(self, frame: np.ndarray, text: str, pos: Tuple[int, int], 
                         bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int] = (255, 255, 255),
                         font_scale: float = 0.7, thickness: int = 2) -> None:
        """Render text with background box"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = pos
        cv2.rectangle(frame, (x - 10, y - h - 10), (x + w + 10, y + baseline + 10), bg_color, -1)
        cv2.putText(frame, text, pos, font, font_scale, text_color, thickness)
    
    def _render_status_bar(self, frame: np.ndarray, fps: float) -> None:
        """Render bottom status bar"""
        h = frame.shape[0]
        status = f"FPS: {fps:.1f} | Press 'H' for Help | 'Q' to Quit"
        cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _render_gesture_label(self, frame: np.ndarray, gesture: Gesture) -> None:
        """Render current gesture name"""
        if gesture not in [Gesture.NO_HAND, Gesture.UNKNOWN]:
            label = gesture.value.replace("_", " ").title()
            self._render_text_box(frame, label, (10, 40), (50, 50, 50), (0, 255, 255), 0.8, 2)
    
    def _render_action_message(self, frame: np.ndarray) -> None:
        """Render action confirmation message"""
        self._render_text_box(frame, self.overlay_text, (10, 90), (0, 180, 0), (255, 255, 255), 0.9, 2)
    
    def _render_help_menu(self, frame: np.ndarray) -> None:
        """Render help menu overlay"""
        help_items = [
            "═══ GESTURES ═══",
            "✊ Fist → Play/Pause",
            "👍 Thumbs Up → Volume ↑",
            "👎 Thumbs Down → Volume ↓",
            "☝ Index → Next Video",
            "🤙 Pinky → Previous Video",
            "✌ Peace → Skip +10s",
            "🤟 Three → Skip -10s",
            "✋ Open Palm → Fullscreen",
            "👌 OK → Mute/Unmute",
            "",
            "═══ KEYBOARD ═══",
            "Q → Quit",
            "R → Reset",
            "H → Toggle Help"
        ]
        
        y_start = 30
        for i, text in enumerate(help_items):
            color = (100, 200, 255) if "═══" in text else (255, 255, 255)
            cv2.putText(frame, text, (frame.shape[1] - 320, y_start + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


class GestureController:
    """
    Main application controller integrating gesture recognition and YouTube control.
    Manages the complete pipeline from camera input to video control.
    """
    
    def __init__(self, chromedriver_path: str):
        self.config = Config()
        self.youtube = YouTubeController(chromedriver_path)
        self.recognizer = GestureRecognizer(self.config)
        self.overlay = OverlayRenderer(self.config)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_action_time = 0.0
        self.last_gesture = Gesture.NO_HAND
        self.gesture_stability_count = 0
        
        # Performance tracking
        self.frame_times = []
        self.fps = 0.0
    
    def _initialize_camera(self, camera_id: int = 0) -> None:
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to access camera")
        
        logger.info("✓ Camera initialized")
    
    def _update_fps(self) -> None:
        """Calculate and update FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]
        self.fps = len(self.frame_times)
    
    def _process_gesture_action(self, gesture: Gesture) -> None:
        """Execute YouTube control action based on detected gesture"""
        now = time.time()
        
        # Apply cooldown and stability requirements
        if (now - self.last_action_time < self.config.GESTURE_COOLDOWN or
            gesture in [Gesture.UNKNOWN, Gesture.NO_HAND]):
            return
        
        # Gesture stability tracking
        if gesture == self.last_gesture:
            self.gesture_stability_count += 1
        else:
            self.gesture_stability_count = 1
            self.last_gesture = gesture
        
        if self.gesture_stability_count < self.config.GESTURE_STABILITY_FRAMES:
            return
        
        # Action mapping
        action_map = {
            Gesture.FIST: ("Play/Pause", self.youtube.play_pause),
            Gesture.THUMBS_UP: ("Volume Up", self.youtube.volume_up),
            Gesture.THUMBS_DOWN: ("Volume Down", self.youtube.volume_down),
            Gesture.INDEX: ("Next Video", self.youtube.next_video),
            Gesture.PINKY: ("Previous Video", self.youtube.prev_video),
            Gesture.PEACE: ("Skip Forward", lambda: self.youtube.skip_forward(10)),
            Gesture.THREE: ("Skip Backward", lambda: self.youtube.skip_backward(10)),
            Gesture.OPEN_PALM: ("Fullscreen", self.youtube.toggle_fullscreen),
            Gesture.OK: ("Mute/Unmute", self.youtube.mute_unmute),
        }
        
        if gesture in action_map:
            label, action = action_map[gesture]
            result = action()
            message = result if result else label
            self.overlay.set_message(message)
            
            self.last_action_time = now
            self.gesture_stability_count = 0
    
    def run(self) -> None:
        """Main application loop"""
        try:
            self._initialize_camera()
            self.youtube.open_youtube()
            
            logger.info("🚀 Hand Gesture YouTube Controller started")
            logger.info("Press 'H' for help | 'Q' to quit")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Camera read failed")
                    break
                
                frame = cv2.flip(frame, 1)
                self._update_fps()
                
                # Process hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.recognizer.hands.process(rgb_frame)
                
                gesture = Gesture.NO_HAND
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.recognizer.draw_landmarks(frame, hand_landmarks)
                        gesture = self.recognizer.recognize_gesture(hand_landmarks.landmark)
                
                self._process_gesture_action(gesture)
                self.overlay.render(frame, gesture, self.fps)
                
                # Display frame
                display_frame = cv2.resize(frame, (self.config.DISPLAY_WIDTH, self.config.DISPLAY_HEIGHT))
                cv2.imshow("Hand Gesture YouTube Controller", display_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    self.overlay.toggle_help()
                elif key == ord('r'):
                    self._reset_state()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _reset_state(self) -> None:
        """Reset gesture detection state"""
        self.last_action_time = 0
        self.gesture_stability_count = 0
        self.overlay.set_message("🔄 System Reset")
    
    def _cleanup(self) -> None:
        """Release all resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.youtube.close()
        logger.info("✓ Application closed successfully")


def main():
    """Application entry point"""
    # TODO: Update this path to your ChromeDriver location
    CHROMEDRIVER_PATH = r"C:\path\to\chromedriver.exe"
    
    print("=" * 60)
    print("  Hand Gesture YouTube Controller v2.0")
    print("  Real-time gesture recognition for YouTube control")
    print("=" * 60)
    
    try:
        controller = GestureController(CHROMEDRIVER_PATH)
        controller.run()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Update CHROMEDRIVER_PATH in the script")
        print("  2. Ensure ChromeDriver matches your Chrome version")
        print("  3. Check camera permissions")
        print("  4. Install requirements: pip install opencv-python mediapipe selenium")


if __name__ == "__main__":
    main()
