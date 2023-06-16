import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

# 얼굴 감지 모델 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 모델 클래스 정의 (예시)


# 모델 불러오기
model =torch.load('best_model.pth')

# 분류할 라벨 정의
labels = ['label1', 'label2', 'label3']  # 분류할 라벨을 적절히 수정해주세요.

# 웹캠 열기
cap =cv2.VideoCapture(cv2.CAP_DSHOW+0) #0번이 내장카메라, 1번이 외장카메라

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    # 이미지를 회색으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 감지된 얼굴에 대해 반복
    for (x, y, w, h) in faces:
        # 얼굴 이미지 추출
        face_img = gray[y:y + h, x:x + w]

        # 이미지 크기 조정 및 정규화
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        face_img = transform(face_img)
        face_img = Variable(face_img.unsqueeze(0))

        # 얼굴 분류
        output = model(face_img)
        _, predicted = torch.max(output.data, 1)
        label_index = predicted.item()
        label = labels[label_index]

        # 얼굴에 사각형 그리기 및 분류 결과 텍스트 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 프레임 표시
    cv2.imshow('Face Classification', frame)

    # 'q' 키를 누르면 종료
