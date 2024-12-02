import cv2
import numpy as np
import torch
from mss import mss

# YOLOv5n 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# 스크린 캡처 객체 생성
sct = mss()
monitor = sct.monitors[1]  # 첫 번째 모니터 선택 (필요 시 [2], [3]으로 변경)

while True:
    # 화면 캡처
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)  # numpy 배열로 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # BGRA에서 BGR로 변환

    # 객체 감지
    results = model(frame)

    # 결과를 프레임에 그리기
    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] == 'person' and conf >= 0.2:  # 신뢰도 임계값 조정
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 프레임 크기 조정 (작게)
    small_frame = cv2.resize(frame, (800, 450))  # (너비, 높이) 조정

    # 프레임 출력
    cv2.imshow('Screen Intrusion Detection', small_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cv2.destroyAllWindows()

