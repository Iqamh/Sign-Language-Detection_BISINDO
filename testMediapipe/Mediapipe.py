import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cv2.namedWindow("Hand Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Control", 1280, 720)

def is_iloveyou_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    return (thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and
            index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)

def is_thumbs_up(hand_landmarks):
    return (hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y and
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[12].y and
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[16].y and
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[20].y)

def is_peace_sign(hand_landmarks):
    return (hand_landmarks.landmark[8].y < hand_landmarks.landmark[16].y and
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[20].y)

def is_fist(hand_landmarks):
    return (hand_landmarks.landmark[4].y > hand_landmarks.landmark[2].y and
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and
            hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y)

def is_open_hand(hand_landmarks):
    return (hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y and
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y and
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y and
            hand_landmarks.landmark[16].y < hand_landmarks.landmark[13].y and
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[17].y)

def is_pointing_up(hand_landmarks):
    return (hand_landmarks.landmark[8].y < hand_landmarks.landmark[4].y and
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[12].y and
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[16].y and
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[20].y)

def is_okay_sign(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return (abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05)

def draw_menu(image, x, y):
    button_width, button_height = 150, 50
    cv2.rectangle(image, (x, y), (x + button_width, y + button_height), (0, 255, 0), -1)
    cv2.rectangle(image, (x, y + button_height + 10), (x + button_width, y + 2 * button_height + 10), (255, 0, 0), -1)
    cv2.putText(image, 'Button 1', (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Button 2', (x + 10, y + button_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return (x, y, button_width, button_height)

def check_button_hover(hand_landmarks, button_coords, frame, clicked_buttons):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    height, width, _ = frame.shape
    x, y = int(index_tip.x * width), int(index_tip.y * height)
    
    x1, y1, bw, bh = button_coords
    if x1 < x < x1 + bw and y1 < y < y1 + bh and not clicked_buttons[0]:
        print("Button 1 Pressed")
        clicked_buttons[0] = True
        return (255, 0, 0)  # Blue color in BGR format
    elif x1 < x < x1 + bw and y1 + bh + 10 < y < y1 + 2 * bh + 10 and not clicked_buttons[1]:
        print("Button 2 Pressed")
        clicked_buttons[1] = True
        return (0, 0, 255)  # Red color in BGR format
    return None

def draw_hand_landmarks(image, hand_landmarks, landmark_color, connection_color):
    for landmark in hand_landmarks.landmark:
        h, w, _ = image.shape
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, landmark_color, -1)
    
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_landmark = hand_landmarks.landmark[start_idx]
        end_landmark = hand_landmarks.landmark[end_idx]
        start_x, start_y = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
        end_x, end_y = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])
        cv2.line(image, (start_x, start_y), (end_x, end_y), connection_color, 2)

def main():
    cap = cv2.VideoCapture(0)
    menu_displayed = False
    menu_coords = None
    clicked_buttons = [False, False]
    landmark_color = (0, 255, 0)  # Default color: green
    connection_color = (255, 255, 255)  # Default connection color: white

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    color = None

                    if is_thumbs_up(hand_landmarks):
                        cv2.putText(image, 'Thumbs Up!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    elif is_peace_sign(hand_landmarks):
                        cv2.putText(image, 'Peace Sign!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
                    elif is_fist(hand_landmarks):
                        cv2.putText(image, 'Fist!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    elif is_open_hand(hand_landmarks):
                        cv2.putText(image, 'Open Hand!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
                    elif is_pointing_up(hand_landmarks):
                        cv2.putText(image, 'Pointing Up!', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                    elif is_okay_sign(hand_landmarks):
                        cv2.putText(image, 'Okay Sign!', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
                    elif is_iloveyou_gesture(hand_landmarks):
                        cv2.putText(image, 'I Love You!', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                        if not menu_displayed:
                            menu_coords = draw_menu(image, 200, 100)
                            menu_displayed = True
                    
                    if menu_displayed and menu_coords:
                        color = check_button_hover(hand_landmarks, menu_coords, frame, clicked_buttons)
                    
                    if color:
                        landmark_color = color
                        connection_color = color

                    draw_hand_landmarks(image, hand_landmarks, landmark_color, connection_color)
            
            cv2.imshow('Hand Control', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
