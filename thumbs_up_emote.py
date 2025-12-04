import cv2
import mediapipe as mp
import time
import os
import subprocess
from PIL import Image
import numpy as np

GIF_VIEWER_PROCESS_NAME = "Preview"  # or "Safari", etc. depending on what opens the GIF

audio_process_open = False
audio_proc = None


downgif_playing = False
downgif_frame_idx = 0
downgif_next_time = 0.0

down_audio_process_open = False
down_audio_proc = None

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For reference
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2

INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


def is_thumb_up(hand_landmarks):
    """
    Very rough "thumbs up" heuristic:
    - Thumb vertical: tip above other thumb joints
    - Other fingers folded: tip below PIP joint
    Works best when your thumbs are clearly pointed up
    and other fingers curled in.
    """
    lm = hand_landmarks.landmark

    # Thumb extended upwards (y is smaller when higher on screen)
    thumb_extended = (
        lm[THUMB_TIP].y < lm[THUMB_IP].y < lm[THUMB_MCP].y
    )

    # Other fingers should be folded (tip below PIP)
    fingers_folded = (
        lm[INDEX_TIP].y > lm[INDEX_PIP].y and
        lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y and
        lm[RING_TIP].y > lm[RING_PIP].y and
        lm[PINKY_TIP].y > lm[PINKY_PIP].y
    )

    return thumb_extended and fingers_folded

def is_thumb_down(hand_landmarks):
    """
    Very rough "thumbs down" heuristic:
    - Thumb vertical: tip below other thumb joints (since y increases downward)
    - Other fingers folded: tip below PIP joint
    """
    lm = hand_landmarks.landmark

    # Thumb extended downwards (y is larger when lower on screen)
    thumb_extended_down = (
        lm[THUMB_TIP].y > lm[THUMB_IP].y > lm[THUMB_MCP].y
    )

    # Other fingers should be folded (same as thumbs up)
    fingers_folded = (
        lm[INDEX_TIP].y > lm[INDEX_PIP].y and
        lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y and
        lm[RING_TIP].y > lm[RING_PIP].y and
        lm[PINKY_TIP].y > lm[PINKY_PIP].y
    )

    return thumb_extended_down and fingers_folded

def main():
    global audio_process_open
    global audio_proc
    global down_audio_process_open
    global down_audio_proc
    global downgif_playing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check permissions / device index.")

    # Emote state
    emote_active = False
    emote_until = 0.0
    EMOTE_DURATION = 2.0  # seconds to show emote
    COOLDOWN = 1.0        # seconds between triggers
    last_trigger_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        
        # Load emote image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # OpenCV often has issues with GIF files, so we'll assume a PNG is used instead.
        # The user should convert '200w.gif' to '200w.png' and place it in the 'emotes' directory.
        emote_path = os.path.join(script_dir, "emotes", "200w.gif")
        if not os.path.exists(emote_path):
            raise RuntimeError(f"Could not find emote GIF at {emote_path}")
        
        # Audio path
        audio_path = os.path.join(script_dir, "audio", "heheheha.mp3")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Could not find audio file at {audio_path}")
        
        # Load thumbs-down emote GIF
        downgif_path = os.path.join(script_dir, "emotes", "cry.gif")
        if not os.path.exists(downgif_path):
            raise RuntimeError(f"Could not find cry GIF at {downgif_path}")

        # Load thumbs-down audio
        down_audio_path = os.path.join(script_dir, "audio", "gobcry.mp3")
        if not os.path.exists(down_audio_path):
            raise RuntimeError(f"Could not find gobcry audio at {down_audio_path}")
        
        # --- Load thumbs-down GIF frames ---
        downgif = Image.open(downgif_path)
        downgif_frames = []
        downgif_durations = []

        try:
            while True:
                frame_pil = downgif.convert("RGBA")
                downgif_frames.append(frame_pil.copy())
                downgif_durations.append(downgif.info.get("duration", 100) / 1000.0)
                downgif.seek(downgif.tell() + 1)
        except EOFError:
            pass

        downgif_frames_cv = []
        for f in downgif_frames:
            arr = np.array(f)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            downgif_frames_cv.append(arr)

        if not downgif_frames_cv:
            raise RuntimeError("cry.gif has no frames.")

        # Load GIF frames
        gif = Image.open(emote_path)
        gif_frames = []
        gif_durations = []

        gif_playing = False
        gif_frame_idx = 0
        gif_next_time = 0.0

        try:
            while True:
                frame_pil = gif.convert("RGBA")  # keep alpha
                gif_frames.append(frame_pil.copy())
                # duration in ms -> seconds, default 100ms
                gif_durations.append(gif.info.get("duration", 100) / 1000.0)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Convert frames to OpenCV BGRA numpy arrays
        gif_frames_cv = []
        for f in gif_frames:
            arr = np.array(f)
            # RGBA -> BGRA
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            gif_frames_cv.append(arr)

        if not gif_frames_cv:
            raise RuntimeError("GIF has no frames.")
        
        

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip for more natural selfie view
            frame = cv2.flip(frame, 1)

            # Convert to RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            thumbs_up_count = 0
            thumbs_down_count = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks for debugging
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    if is_thumb_up(hand_landmarks):
                        thumbs_up_count += 1
                    elif is_thumb_down(hand_landmarks):
                        thumbs_down_count += 1

            # If both hands are thumbs up, trigger emote
            now = time.time()

            # If both hands are thumbs up, play GIF in separate window
            if thumbs_up_count >= 2:
                if not gif_playing:
                    gif_playing = True
                    gif_frame_idx = 0
                    gif_next_time = now
                    cv2.namedWindow("Emote", cv2.WINDOW_AUTOSIZE)

                        # If audio is NOT playing → start it
                if not audio_process_open:
                    audio_proc = subprocess.Popen(["afplay", audio_path])
                    audio_process_open = True
                    print("Started heheheha.mp3")

                    # If audio *finished*, restart it
                elif audio_proc.poll() is not None:  
                    audio_proc = subprocess.Popen(["afplay", audio_path])
                    print("Looping heheheha.mp3")

                if now >= gif_next_time:
                    gif_frame_idx = (gif_frame_idx + 1) % len(gif_frames_cv)
                    gif_next_time = now + gif_durations[gif_frame_idx]

                # show current frame
                cv2.imshow("Emote", gif_frames_cv[gif_frame_idx])

            elif thumbs_down_count >= 2:
                # Start GIF window if not running
                if not downgif_playing:
                    downgif_playing = True
                    downgif_frame_idx = 0
                    downgif_next_time = now
                    cv2.namedWindow("DownEmote", cv2.WINDOW_AUTOSIZE)

                # Start audio if not running
                if not down_audio_process_open:
                    down_audio_proc = subprocess.Popen(["afplay", down_audio_path])
                    down_audio_process_open = True
                    print("Started gobcry.mp3")

                # Restart audio if it finished
                elif down_audio_proc.poll() is not None:
                    down_audio_proc = subprocess.Popen(["afplay", down_audio_path])
                    print("Looping gobcry.mp3")

                # Advance GIF frames
                if now >= downgif_next_time:
                    downgif_frame_idx = (downgif_frame_idx + 1) % len(downgif_frames_cv)
                    downgif_next_time = now + downgif_durations[downgif_frame_idx]

                cv2.imshow("DownEmote", downgif_frames_cv[downgif_frame_idx])
                
                
             # If thumbs are NOT up
            else:
                # Stop audio if currently playing
                if audio_process_open and audio_proc is not None:
                    try:
                        audio_proc.terminate()
                        print("Stopped heheheha.mp3")
                    except Exception as e:
                        print("Error stopping audio:", e)

                    audio_proc = None
                    audio_process_open = False
                if gif_playing:
                    gif_playing = False
                    cv2.destroyWindow("Emote")
                # No thumbs up → close GIF window if it was open
                # Stop thumbs-down GIF/audio
                if down_audio_process_open and down_audio_proc is not None:
                    try:
                        down_audio_proc.terminate()
                        print("Stopped gobcry.mp3")
                    except Exception as e:
                        print("Error stopping thumbs-down audio:", e)

                    down_audio_proc = None
                    down_audio_process_open = False

                if downgif_playing:
                    downgif_playing = False
                    cv2.destroyWindow("DownEmote")
                

            

            cv2.putText(
                frame,
                f"Thumbs up hands: {thumbs_up_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Thumbs down hands: {thumbs_down_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Double Thumbs Up -> Clash Emote (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()