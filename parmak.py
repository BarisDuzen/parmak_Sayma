import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# modüllerin tanımlanması
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# el nesnemizi tanımlıyoruz
hands = mp_hand.Hands()

# Parmak uçlarının landmark ID'leri
tipIds = [4, 8, 12, 16, 20]

while True:
    liste = []
    
    success, frame = cap.read()
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_img)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = frame.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                liste.append([id, cx, cy, lm.z])  # z koordinatını da ekledik
                #Burada her bir karede elimizin koordinatları ve id si listeye atılıyor.
    
    # Eğer landmark verisi varsa parmak durumlarını kontrol et
    if len(liste) != 0:
        
        fingers = []
        
        if liste[tipIds[0]][1]>liste[tipIds[0]-2][1]:#Baş parmak kontrolü yapılıyor.Ancak sağ elin için geçerli.
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            # Eğer parmak ucu, parmağın altındaki landmark'tan yukarıda ise parmak açık
            if liste[tipIds[id]][2] < liste[tipIds[id] - 2][2]: #Burada her parmağın açık olup olmadığı sorgulanıyor.
                fingers.append(1)  # Parmak açık
            else:
                fingers.append(0)  # Parmak kapalı
                
        cv2.putText(frame,"Parmak sayisi:"+str(fingers.count(1)), (10,75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,0))
    
    cv2.imshow("goruntu", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşuna basıldığında çık
        break

cap.release()
cv2.destroyAllWindows()
