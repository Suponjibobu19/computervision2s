""
import cv2
import mediapipe as mp
import numpy as np
import time
import rtmidi

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam non accessible")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MIDI Setup
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()

if available_ports:
    midiout.open_port(0)
else:
    midiout.open_virtual_port("Virtual MIDI")

# Notes pour chaque main (transposées)
notes_gauche_freq = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63]  # Do3 → Do4
notes_droite_freq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # Do4 → Do5
notes_noms = ["Do", "Ré", "Mi", "Fa", "Sol", "La", "Si", "Do(haut)"]
nb_notes = len(notes_noms)

wave_dict = {1: 0, 2: 24, 3: 40, 4: 56, 5: 115}  # MIDI Program Change
active_notes = [None, None]  # [main gauche, main droite]
active_programs = [None, None]

print("Musique gestuelle via Qsynth/MIDI - Échap pour quitter.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Enlever effet miroir
    height, width, _ = frame.shape
    mid_x = width // 2
    colonne_largeur = width // 2 // nb_notes

    # Dessin des deux claviers
    for i in range(1, nb_notes):
        # Clavier gauche
        x_pos_g = i * colonne_largeur
        cv2.line(frame, (x_pos_g, 0), (x_pos_g, height), (100, 100, 100), 1)
        # Clavier droit
        x_pos_d = mid_x + i * colonne_largeur
        cv2.line(frame, (x_pos_d, 0), (x_pos_d, height), (100, 100, 100), 1)

    for i in range(nb_notes):
        cv2.putText(frame, notes_noms[i], (i * colonne_largeur + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        cv2.putText(frame, notes_noms[i], (mid_x + i * colonne_largeur + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" ou "Right"
            hand_idx = 0 if label == "Left" else 1

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

            x_norm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            y_norm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            x_px = int(x_norm * width)
            y_px = int(y_norm * height)
            cv2.circle(frame, (x_px, y_px), 10, (0, 255, 0), -1)

            # Calcul index de note selon la zone
            if hand_idx == 0:  # Main gauche
                zone_x = x_px if x_px < mid_x else 0
                idx = int(zone_x / colonne_largeur)
                notes = notes_gauche_freq
            else:  # Main droite
                zone_x = x_px - mid_x if x_px >= mid_x else 0
                idx = int(zone_x / colonne_largeur)
                notes = notes_droite_freq

            idx = min(max(idx, 0), nb_notes - 1)
            base_freq = notes[idx]
            factor = 0.5 + (1 - y_norm) * 1.5
            freq = base_freq * factor
            midi_note = int(69 + 12 * np.log2(freq / 440.0))

            lm = hand_landmarks.landmark
            fingers_up = 0
            if lm[mp_hands.HandLandmark.THUMB_TIP].x > lm[mp_hands.HandLandmark.THUMB_IP].x:
                fingers_up += 1
            tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]
            pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.PINKY_PIP]
            for tip, pip in zip(tips, pips):
                if lm[tip].y < lm[pip].y:
                    fingers_up += 1

            program = wave_dict.get(fingers_up, 0)

            if active_programs[hand_idx] != program:
                midiout.send_message([0xC0 + hand_idx, program])
                active_programs[hand_idx] = program

            if active_notes[hand_idx] != midi_note:
                if active_notes[hand_idx] is not None:
                    midiout.send_message([0x80 + hand_idx, active_notes[hand_idx], 0])
                midiout.send_message([0x90 + hand_idx, midi_note, 100])
                active_notes[hand_idx] = midi_note

            info = f"{label} hand: {notes_noms[idx]} MIDI:{midi_note} | Doigts:{fingers_up} | Instr:{program}"
            cv2.putText(frame, info, (10, height - 20 - hand_idx*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        for i in range(2):
            if active_notes[i] is not None:
                midiout.send_message([0x80 + i, active_notes[i], 0])
                active_notes[i] = None

    cv2.imshow("Musique gestuelle - Qsynth/MIDI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
midiout.close_port()
