import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Đọc dữ liệu từ file 'Iris.csv'
df = pd.read_csv('iris.csv')

# In tên các cột để kiểm tra dữ liệu
print(df.columns)

# Mã hóa cột 'species' thành các giá trị số
df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Kiểm tra lại dữ liệu sau khi mã hóa
print(df.head())

# Chọn các đặc trưng (features) và nhãn (target)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Các đặc trưng
y = df['species']  # Nhãn (loài hoa đã được mã hóa thành số)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình Logistic Regression và huấn luyện
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Lưu mô hình vào file 'model.pkl'
pickle.dump(model, open('model.pkl', 'wb'))

print("Mô hình đã được lưu vào 'model.pkl'.")