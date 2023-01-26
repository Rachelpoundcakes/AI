# Scikit-Learn에서는 데이터 스플릿이 필요했지만 Keras에는 이미 쪼개져 있으므로 그냥 가져다 쓰면 된다. 
from keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000
print(test_images.shape) # (10000, 28, 28)

from keras import models
from keras import layers

# Sequential: 순차적으로 신경망을 만든다는 뜻
# 2개짜리 layer를 만들어 보자.
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) # 넉넉하게 512를 준다.
network.add(layers.Dense(10, activation='softmax')) # 출력값은 0~9 중 하나이므로 10을 준다.
# ==> 여기까자 신경망 추가! 아직 완성은 X

# optimizer 최적화 방법에 따라 접근 방법이 달라진다.
# 뭘 써야 할지 모를 땐 rmsprop를 쓰자
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 데이터 타입의 변환 28*28=784
# train 될 수 있도록 28*28픽셀의 이미지를 하나로 쭉 늘어놓는다(reshape)
train_images = train_images.reshape((60000, 28*28)) # uint8 음수가 없는 정수 8비트 타입
train_images = train_images.astype('float32') # float32로 바꿔준다.
print(train_images[0])

# 255로 나누어 소수점으로 바꿔준다. tensorflow가 사용하기 용이한 형태이다.
train_images = train_images.astype('float32') /255

# 테스트 이미지도 마찬가지로 소수점으로 바꿔준다.
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 0~9까지 정해져 있는 값이 나오도록 categorical 데이터라는 것을 명시한다.
# 분류형 데이터의 설정
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# ==> 데이터 준비 완료 이제 학습시켜보자

network.fit(train_images, train_labels, epochs=5, batch_size=128)
# loss는 오차. 오차는 점점 줄어든다. 과대적합을 막으려면 어느 시점에서 멈춰야 한다.

# 평가 결과 보기
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
"""
Epoch 1/5
469/469 [==============================] - 3s 5ms/step - loss: 0.2650 - accuracy: 0.9238
Epoch 2/5
469/469 [==============================] - 2s 5ms/step - loss: 0.1072 - accuracy: 0.9675
Epoch 3/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0703 - accuracy: 0.9793
Epoch 4/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0509 - accuracy: 0.9847
Epoch 5/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0386 - accuracy: 0.9886
313/313 [==============================] - 1s 3ms/step - loss: 0.0717 - accuracy: 0.9791
test_acc:  0.9790999889373779
"""