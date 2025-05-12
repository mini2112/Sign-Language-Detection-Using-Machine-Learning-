import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

expected_landmarks = 21 * 2  # 21 landmarks, each with (x, y)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # ✅ Ensure all samples have the same size
            if len(data_aux) == expected_landmarks:
                data.append(data_aux)
                labels.append(int(dir_))

# Save dataset only if all entries are of the same shape
if len(set(len(d) for d in data)) == 1:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("✅ Dataset created successfully!")
else:
    print("❌ Error: Inconsistent data shapes detected! Try recollecting the dataset.")
