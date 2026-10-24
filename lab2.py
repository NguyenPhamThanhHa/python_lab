#Khai báo thư viện & Tải dữ liệu
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit

# Tải dữ liệu
X, Y = load_coffee_data()
print(X.shape, Y.shape)

# Vẽ biểu đồ dữ liệu rang cà phê
plt_roast(X, Y)

#Chuẩn hóa dữ liệu (Normalize Data)
# Tạo lớp chuẩn hóa và học giá trị mean, variance từ dữ liệu
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  
Xn = norm_l(X)

# Nhân bản dữ liệu để tăng kích thước tập huấn luyện
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))

#xây dựng kiến trúc mạng
tf.random.set_seed(1234)  # Đảm bảo kết quả nhất quán
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation='sigmoid', name='layer1'),
    Dense(1, activation='sigmoid', name='layer2')
])

model.summary()

#biên dịch và huấn luyện mô hình
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(Xt, Yt, epochs=10)

#đưa ra dự đoán
X_test = np.array([
    [200, 13.9],  # Ví dụ mẫu dương (good roast)
    [200, 17]     # Ví dụ mẫu âm (bad roast)
])

X_testn = norm_l(X_test)
predictions = model.predict(X_testn)

# Chuyển đổi xác suất thành quyết định dựa trên ngưỡng 0.5
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

#Phân tích chức năng của các Lớp (Layer Functions)
# Vẽ đầu ra của từng đơn vị trong lớp 1
plt_layer(X, Y.reshape(-1,), W1, b1, norm_l)

# Vẽ đầu ra của lớp cuối cùng
plt_output_unit(W2, b2)

# Vẽ toàn bộ mạng neural
netf = lambda x: model.predict(norm_l(x))
plt_network(X, Y, netf)