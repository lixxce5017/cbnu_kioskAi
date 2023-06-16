import os
import torch
import torchvision
from PIL.Image import fromarray
from torchvision.models import ResNet18_Weights

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU 할당
print(device)
from PIL import Image
import glob
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torch.optim import optimizer
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image, display
import face_recognition as fr

CFG = {
    'IMG_SIZE': 128,  # 이미지 사이즈128
    'EPOCHS': 100,  # 에포크
    'BATCH_SIZE': 16,  # 배치사이즈
    'SEED': 1,  # 시드
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])  # 시드고정 함수 선언후 시드고정
train_path = sorted(glob.glob("C:/Users/lixxc/Desktop/original/*.jpg"))

print(len(train_path))  # 총 데이터 갯수

train_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),  # 각 이미지 같은 크기로 resize
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 평균과 표준편차를 0.5로 정규화
])

test_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),  # 각 이미지 같은 크기로 resize
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 평균과 표준편차를 0.5로 정규화
])
Horizontal_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),  # 각 이미지 같은 크기로 resize
    transforms.RandomHorizontalFlip(1.0),  # Horizontal = 좌우반전
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 평균과 표준편차를 0.5로 정규화
])

remove_idx = [2, 6, 22, 85, 86, 105, 112, 128, 135, 141, 155, 156, 161, 183, 192, 197, 209, 219, 245, 246, 281, 290,
              292, 293, 295, 298, 302, 303, 313, 326, 328, 329, 331, 367, 372, 375, 379, 385, 386, 394, 408, 412, 426,
              430, 436, 437, 438, 443, 455, 457, 460, 461, 470, 481, 482, 483, 512, 517, 523, 536, 538, 558, 563, 579,
              585, 596, 631, 632, 633, 638, 639, 640, 646, 649, 665, 666, 668, 703, 708, 720, 729, 746, 769, 776, 778,
              782, 788, 790, 805, 808, 813, 814, 816, 833, 842, 855, 858, 880, 882, 915, 916, 919, 962, 966, 968, 972,
              974, 988, 990, 999, 1006, 1022, 1027, 1040, 1088, 1119, 1120, 1133, 1141, 1159, 1171, 1189, 1195, 1210,
              1212, 1215, 1234, 1240, 1262, 1280, 1288, 1300, 1308, 1329, 1332, 1357, 1374, 1375, 1380, 1386, 1472,
              1508, 1534, 1540, 1570, 1588, 1593, 1602, 1609, 1614, 1625, 1659, 1665, 1676, 1692, 1722, 1730, 1739,
              1757, 1778, 1781, 1785, 1829, 1834, 1840, 1854, 1857, 1876, 1884, 1889, 1894, 1896, 1915, 1957, 1984,
              2015, 2030, 2042, 2047, 2048, 2077, 2083, 2105, 2111, 2114, 2118, 2138, 2156, 2231, 2236, 2256, 2274,
              2292, 2351, 2403, 2407, 2416, 2420, 2427, 2443, 2447, 2462, 2503, 2510, 2544, 2552, 2557, 2576, 2579,
              2594, 2597, 2604, 2607, 2637, 2643, 2658, 2738, 2744, 2758, 2791, 2799, 2835, 2854, 2870, 2961, 2977,
              3028, 3048, 3087, 3100, 3129, 3131, 3136, 3137, 3143, 3144, 3150, 3162, 3198, 3210, 3224, 3236, 3245,
              3329, 3335, 3336, 3369, 3373, 3381, 3407, 3473, 3491, 3496, 3530, 3553, 3565, 3574, 3580, 3582, 3594,
              3599, 3610, 3619, 3633, 3679, 3683, 3689, 3699, 3708, 3712, 3713, 3717, 3725, 3757, 3763, 3774, 3791,
              3799, 3833, 3870, 3874, 3933, 3936, 3949, 3970, 3981, 3994, 4009, 4015, 4033, 4038, 4093, 4129, 4146,
              4147, 4149, 4172, 4174, 4182, 4185, 4189, 4210, 4256, 4260, 4296, 4298, 4304, 4369, 4371, 4376, 4414,
              4511, 4514, 4519, 4532, 4551, 4558, 4577, 4584, 4598, 4612, 4613, 4624, 4663, 4672, 4682, 4685, 4687,
              4696, 4725, 4735, 4745, 4760, 4774, 4785, 4787, 4788, 4806, 4818, 4831, 4834, 4843, 4848, 4858, 4859,
              4873, 4874, 4883, 4886, 4897, 4900, 4901, 4910, 4915, 4945, 4950, 4952, 4956, 4960, 4964, 5000, 5001,
              5002, 5014, 5020, 5023, 5025, 5042, 5044, 5084, 5094, 5150, 5158, 5168, 5266, 5278, 5284, 5295, 5331,
              5333, 5351, 5354, 5359, 5361, 5365, 5369, 5416, 5437, 5441, 5457, 5477, 5480, 5483, 5518, 5540, 5547,
              5561, 5570, 5587, 5591, 5592, 5607, 5622, 5623, 5624, 5643, 5659, 5667, 5687, 5699, 5701, 5708, 5709,
              5720, 5732, 5755, 5757, 5759, 5763, 5772, 5780, 5785, 5788, 5801, 5811, 5833, 5859, 5861, 5862, 5910,
              5920, 5932, 5949, 5961, 5999, 6002, 6019, 6036, 6060, 6062, 6063, 6065, 6069, 6074, 6133, 6168, 6183,
              6224, 6225, 6230, 6235, 6260, 6263, 6269, 6273, 6302, 6306, 6313, 6334, 6344, 6354, 6362, 6368, 6372,
              6384, 6406, 6415, 6425, 6433, 6440, 6441, 6442, 6446, 6447, 6449, 6455, 6459, 6469, 6474, 6475, 6492,
              6496, 6499, 6500, 6505, 6506, 6523, 6525, 6570, 6576, 6579, 6606, 6614, 6615, 6616, 6628, 6645, 6656,
              6660, 6668, 6675, 6678, 6702, 6728, 6731, 6743, 6766, 6779, 6783, 6786, 6790, 6803, 6816, 6831, 6836,
              6837, 6854, 6868, 6888, 6900, 6901, 6905, 6925, 6943, 6959, 6960, 6965, 6971, 6972, 6973, 6974, 6975,
              6978, 6986, 7004, 7006, 7010, 7012, 7016, 7021, 7022, 7038, 7053, 7063, 7072, 7082, 7083, 7085, 7090,
              7092, 7093, 7100, 7116, 7120, 7128, 7155, 7159, 7165, 7166, 7171, 7181, 7182, 7195, 7199, 7202, 7203,
              7206, 7208, 7212, 7222, 7226, 7229, 7240, 7245, 7251, 7256, 7258, 7266, 7267, 7275, 7287, 7289, 7312,
              7315, 7328, 7329, 7340, 7344, 7354, 7355, 7356, 7357, 7359, 7360, 7362, 7372, 7377, 7387, 7396, 7401,
              7417, 7425, 7426, 7435, 7436, 7446, 7449, 7460, 7461, 7473, 7484, 7485, 7486, 7499, 7515, 7518, 7519,
              7532, 7536, 7548, 7570, 7578, 7585, 7589, 7592, 7596, 7611, 7624, 7626, 7627, 7629, 7633, 7636, 7641,
              7642, 7643, 7645, 7653, 7655, 7659, 7662, 7685, 7686, 7692, 7699, 7712, 7716, 7721, 7731, 7744, 7746,
              7756, 7765, 7766, 7768, 7772, 7773, 7776, 7787, 7788, 7796, 7808, 7810, 7813, 7819, 7820, 7821, 7824,
              7828, 7832, 7843, 7849, 7858, 7860, 7863, 7874, 7889, 7911, 7931, 7940, 7942, 7944, 7950, 7957, 7959,
              7960, 7966, 7969, 7970, 7972, 7973, 7989, 8001, 8002, 8007, 8009, 8011, 8019, 8025, 8034, 8045, 8053,
              8057, 8058, 8076, 8077, 8085, 8086, 8089, 8093, 8103, 8118, 8122, 8166, 8187, 8202, 8203, 8207, 8209,
              8211, 8221, 8234, 8235, 8246, 8252, 8259, 8279, 8291, 8294, 8300, 8301, 8304, 8306, 8353, 8354, 8356,
              8358, 8360, 8364, 8394, 8410, 8415, 8428, 8433, 8443, 8446, 8453, 8457, 8505, 8538, 8548, 8553, 8613,
              8614, 8625, 8626, 8628, 8633, 8634, 8665, 8688, 8708, 8709, 8741, 8751, 8752, 8756, 8763, 8805, 8813,
              8818, 8829, 8833, 8834, 8838, 8867, 8886, 8914, 8915, 8925, 8936, 8938, 8942, 8951, 8967, 8983, 9009,
              9012, 9021, 9050, 9090, 9091, 9093, 9103, 9106, 9145, 9186, 9196, 9203, 9211, 9224, 9254, 9268, 9307,
              9330, 9331, 9334, 9338, 9339, 9356, 9359, 9390, 9407, 9427, 9450, 9454, 9462, 9465, 9468, 9474, 9482,
              9484, 9499, 9512, 9516, 9526, 9532, 9537, 9555, 9556, 9559, 9572, 9574, 9577, 9590, 9647, 9652, 9713,
              9733, 9736, 9738, 9750, 9771, 9814, 9819, 9869, 9888, 9900, 9916, 9920, 9924, 9925, 9935, 9965, 9987,
              9990, 10054, 10066, 10104, 10111, 10117, 10119, 10123, 10143, 10150, 10156, 10160, 10161, 10166, 10186,
              10193, 10198, 10203, 10237, 10282, 10284, 10285, 10300, 10309, 10338, 10343, 10344, 10366, 10375, 10385,
              10400, 10401, 10404, 10411, 10412, 10416, 10422, 10425, 10427, 10434, 10436, 10462, 10470, 10474, 10477,
              10498, 10500, 10514, 10518, 10526, 10534, 10550, 10553, 10565, 10594, 10603, 10626, 10628, 10653, 10690,
              10705, 10710, 10720, 10721, 10729, 10737, 10745, 10775, 10790, 10795, 10797, 10801, 10816, 10823, 10825,
              10828, 10838, 10841, 10851, 10857, 10863, 10864, 10865, 10872, 10879, 10891, 10893, 10905, 10920, 10922,
              10924, 10935, 10950, 10962, 10981, 10990, 10991, 11001, 11008, 11057, 11058, 11076, 11095, 11113, 11146,
              11148, 11181, 11182, 11191, 11211, 11223, 11246, 11258, 11263, 11266, 11272, 11286, 11294, 11314, 11331,
              11336, 11351, 11355, 11358, 11360, 11369, 11374, 11375, 11397, 11399, 11415, 11418, 11428, 11430, 11445,
              11447, 11456, 11493, 11495, 11496, 11501, 11506, 11515, 11521, 11527, 11529, 11543, 11547, 11563, 11568,
              11610, 11615, 11619, 11622, 11629, 11630, 11632, 11634, 11644, 11646, 11647, 11657, 11658, 11678, 11683,
              11684, 11686, 11718, 11730, 11751, 11753, 11757, 11759, 11773, 11776, 11789, 11822, 11828, 11830, 11835,
              11846, 11847, 11848, 11875, 11899, 11903, 11906, 11910, 11917, 11932, 11936, 11937, 11946, 11973, 11974,
              11976, 11977, 11982, 11986, 11993, 12003, 12008, 12011, 12012, 12013, 12014, 12015, 12016, 12017, 12018,
              12020, 12040, 12053, 12060, 12064, 12087, 12089, 12090, 12103, 12111, 12113, 12118, 12122, 12133, 12146,
              12152, 12153, 12155, 12156, 12169, 12171, 12174, 12176, 12178, 12179, 12187, 12191, 12193, 12194, 12198,
              12207, 12226, 12227, 12229, 12231, 12236, 12243, 12246, 12249, 12251, 12262, 12264, 12273, 12274, 12279,
              12297, 12298, 12299, 12300, 12302, 12308, 12315, 12319, 12320, 12328, 12336, 12337, 12346, 12347, 12350,
              12355, 12358, 12371, 12379, 12381, 12397, 12404, 12421, 12429, 12431, 12439, 12441, 12446, 12448, 12452,
              12456, 12468, 12470, 12481, 12488, 12490, 12492, 12493, 12495, 12507, 12517, 12534, 12537, 12544, 12545,
              12553, 12554, 12555, 12556, 12557, 12558, 12559, 12560, 12562, 12566, 12567, 12568, 12569, 12572, 12575,
              12609, 12625, 12628, 12630, 12632, 12653, 12665, 12670, 12677, 12682, 12684, 12711, 12714, 12718, 12741,
              12747, 12756, 12769, 12779, 12794, 12809, 12812, 12819, 12820, 12829, 12833, 12837, 12842, 12859, 12868,
              12872, 12875, 12883, 12898, 12901, 12946, 12952, 12958, 12960, 12961, 12983, 12993, 13002, 13005, 13006,
              13013, 13021, 13022, 13036, 13042, 13072, 13081, 13111, 13117, 13118, 13123, 13125, 13129, 13133, 13138,
              13146, 13150, 13151, 13158, 13177, 13184, 13185, 13208, 13218, 13228, 13229, 13245, 13251, 13256, 13262,
              13266, 13273, 13274, 13279, 13283, 13301, 13310]

# 위의 셀에서 추출한 인덱스 대입

for idx in sorted(remove_idx, reverse=True):
    del train_path[idx]  # 인덱스값으로 경로중 일부 제거


face_list = []
image_list = []
num = 0
i = 0
non_recognition_idx = []
file_name = []
gender_info = []
for id in train_path:
    name = id[38:40]
    print(name)
    file_name.append(name)  # 파일명의 3번째 A와 .사이의 나이정보를 label로 설정
    age = id[32:37]
    gender_info.append(age)  # gender id저장장

label_list = list(map(int, file_name))  # str형태를 int로 변경
gender_id = list(map(int, gender_info))  # str형태를 int로 변경
train_y = label_list

age_label = []
for i in range(2, 81):
    age_label.append(i)

num_age = []
from collections import Counter

result = Counter(train_y)
print(result)
for key in result:
    print(key, result[key])

for i in range(2, 81):
    num_age.append(result[i])
    print(num_age)

    num_gender = []
    male = 0
    female = 0
for value in gender_id:
    if value < 7381:
        female += 1
    elif value > 14516:
        female += 1
    else:
        male += 1

train_y = []
for i in range(len(label_list)):  # 정확도 향상을 위해 나이를 생애주기별 요약
    if label_list[i] < 30:
        train_y.append(0)
    elif label_list[i] < 50:
        train_y.append(1)
    else:
        train_y.append(2)


num_label = []
from collections import Counter

result = Counter(train_y)


for key in result:
    print(key, result[key])

for i in range(0, 3):
    num_label.append(result[i])

import random


train_img_list = face_list# 0.8비율로 trainset과 validationset으로 스플릿
print(type(train_img_list))
train_label_list = train_y
valid_img_list = face_list
valid_label_list = train_y


def make_weights(labels, nclasses):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)

    _, counts = np.unique(labels, return_counts=True)
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1 / counts[cls], weight_arr)
        # 각 클래스의의 인덱스를 산출하여 해당 클래스 개수의 역수를 확률로 할당한다.
        # 이를 통해 각 클래스의 전체 가중치를 동일하게 한다.

    return weight_arr


weights = make_weights(train_label_list, 3)
weights = torch.DoubleTensor(weights)

weights1 = make_weights(valid_label_list, 3)
weights1 = torch.DoubleTensor(weights1)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weights1, len(weights1))


class ageDataset(Dataset):
    def __init__(self, image,label, train=True, transform=None):
        self.transform = transform
        self.img_list = image
        self.label_list=label
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = Image.fromarray(np.uint8(self.img_list[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

original_dataset = ageDataset(image=train_img_list,label=train_label_list, train=True, transform=train_transform) #original dataset 구축
Horizontal_dataset=ageDataset(image=train_img_list,label=train_label_list, train=True, transform=Horizontal_transform) #horizonal dataset 구축
train_dataset=original_dataset+Horizontal_dataset
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers=2,sampler=sampler) #dataloadet 대입



valid_dataset= ageDataset(image=valid_img_list,label=valid_label_list, train=False, transform=test_transform)  #validation custom dataset 구축
valid_loader = DataLoader(valid_dataset, batch_size = CFG['BATCH_SIZE'], num_workers=0,sampler=sampler1) #dataloadet 대입




model1=torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
print(model1)

model1.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 4),
    nn.Softmax()
)
model1 = model1.to(device)

criterion = torch.nn.CrossEntropyLoss()  # loss function으로 crossentropy 설정
optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-4)  # 최적화 함수로 Adam사용
scheduler = None


def train(model, optimizer, train_loader, vali_loader, scheduler, device):  # 학습 함수정의
    model.to(device)  # 모델에 디바이스 할당
    n = len(train_loader)  # 데이터 갯수 파악악

    # Loss Function 정의
    # criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0
    vali_acc = 0
    best_apoch = 0
    val_loss = []
    tr_loss = []
    for epoch in range(1, CFG["EPOCHS"] + 1):  # 에포크 설정
        model.train()  # 모델 학습
        running_loss = 0.0

        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device)  # 배치 데이터
            optimizer.zero_grad()  # 배치마다 optimizer 초기화

            # Data -> Model -> Output
            logit = model(img)  # 예측값 산출
            loss = criterion(logit, label)  # 손실함수 계산

            # 역전파
            loss.backward()  # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
        print('[%d] Train loss: %.10f' % (epoch, running_loss / len(train_loader)))
        tr_loss.append(running_loss / len(train_loader))
        if scheduler is not None:
            scheduler.step()
        # Validation set 평가
        model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  # 4개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.1f}%)\n'.format(vali_loss / len(vali_loader), correct,
                                                                            len(vali_loader.dataset),
                                                                            100 * correct / len(vali_loader.dataset)))
        val_loss.append(vali_loss / len(vali_loader))
        if best_acc < vali_acc:  # early stopping기법 validation의 정확도가 최고치 갱신시 모델 저장장
            best_acc = vali_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       'C:/Users/lixxc/PycharmProjects/pythonProject/best_model.pth')  # 이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')

    return best_acc, tr_loss, val_loss

best_acc,tr_loss,val_loss=train(model1, optimizer, train_loader,valid_loader, scheduler, device) #학습

print("Rmx")