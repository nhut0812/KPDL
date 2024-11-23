from flask import Flask, render_template, request
import pickle
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình đã lưu
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # Khi người dùng truy cập lần đầu, gửi hình ảnh mặc định
    return render_template('index.html', image_file='cover.jpg')


@app.route('/predict', methods=['POST'])
def predict():
        # Nhận dữ liệu từ form nhập liệu
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Dự đoán loài hoa
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)

        # Map kết quả dự đoán sang tên loài hoa và tên hình ảnh
        species_map = {0: ('Setosa', 'setosa.jpg'), 1: ('Versicolor', 'versicolor.jpg'), 2: ('Virginica', 'virginica.jpg')}
        predicted_species, image_file = species_map[prediction[0]]

        return render_template('index.html', prediction_text=f'Iris - {predicted_species}', image_file=image_file)


if __name__ == "__main__":
    app.run(debug=True)