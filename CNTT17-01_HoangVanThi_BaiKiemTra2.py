import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Thư viện sys sử dụng để in ra các kí tự tiếng việt do python không đọc được
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Ẩn cảnh báo FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Đọc file Excel
file_path = "CNTT17-01_HoangVanThi_BaiKiemTra2.xlsx"
df = pd.read_excel(file_path)

# In dữ liệu gốc
print("Dữ liệu gốc:")
print(df.head(50))

# Kiểm tra giá trị thiếu
print("\nSố lượng giá trị thiếu trong mỗi cột:")
print(df.isnull().sum())

# Hiển thị loại dữ liệu theo mô tả dễ hiểu hơn
data_types_desc = {
    "object": "Chuỗi ký tự (Text)",
    "int64": "Số nguyên",
    "float64": "Số thực",
    "datetime64[ns]": "Dữ liệu thời gian"
}

print("\nLoại dữ liệu của từng cột:")
for col in df.columns:
    dtype = str(df[col].dtype)
    print(f"- {col}: {data_types_desc.get(dtype, dtype)}")

# Chuyển đổi cột "Thời gian" về datetime
# Nếu lỗi sẽ trở thành NaT (Not a Time)
df["Thời gian"] = pd.to_datetime(df["Thời gian"], errors='coerce')

# Chuyển các cột số về dạng số nguyên, lỗi sẽ thành NaN
df["Mức ùn tắc"] = pd.to_numeric(df["Mức ùn tắc"], errors='coerce').astype('Int64')
df["Số lượng phương tiện"] = pd.to_numeric(df["Số lượng phương tiện"], errors='coerce').astype('Int64')

# Điền giá trị thiếu bằng trung bình (làm tròn thành số nguyên)
df["Mức ùn tắc"] = df["Mức ùn tắc"].fillna(round(df["Mức ùn tắc"].mean())).astype('Int64')
df["Số lượng phương tiện"] = df["Số lượng phương tiện"].fillna(round(df["Số lượng phương tiện"].mean())).astype('Int64')

# Xóa dòng có thời gian không hợp lệ
df.dropna(subset=["Thời gian"], inplace=True)

# Kiểm tra lại dữ liệu sau khi làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(df.head(50))

# Lưu file mới
cleaned_file_path = "CNTT17-01_HoangVanThi_BaiKiemTra2_cleaned.xlsx"
df.to_excel(cleaned_file_path, index=False)

print(f"Dữ liệu đã làm sạch được lưu tại: {cleaned_file_path}")

# In thống kê cơ bản
print("\nThống kê cơ bản của dữ liệu số:")
print(df.describe())

# Tính toán thêm một số thống kê quan trọng
print("\nThống kê bổ sung:")

# Tính trung bình
print("Trung bình:")
print(df.mean(numeric_only=True))

# Tính trung vị (median)
print("\nTrung vị:")
print(df.median(numeric_only=True))

# Tính phương sai (variance)
print("\nPhương sai:")
print(df.var(numeric_only=True))

# Tính giá trị nhỏ nhất (min)
print("\nGiá trị nhỏ nhất:")
print(df.min(numeric_only=True))

# Tính giá trị lớn nhất (max)
print("\nGiá trị lớn nhất:")
print(df.max(numeric_only=True))


# Vẽ biểu đồ
plt.figure(figsize=(15, 5))

# Biểu đồ cột: Mức ùn tắc và số lượng phương tiện
plt.subplot(1, 3, 1)
sns.barplot(x=df["Mức ùn tắc"], y=df["Số lượng phương tiện"], estimator=sum, palette="Blues")
plt.xlabel("Mức ùn tắc")
plt.ylabel("Tổng số phương tiện")
plt.title("Tổng số phương tiện theo mức ùn tắc")

# Biểu đồ phân phối (Histogram) của số lượng phương tiện
plt.subplot(1, 3, 2)
sns.histplot(df["Số lượng phương tiện"], bins=20, kde=True, color='green')
plt.xlabel("Số lượng phương tiện")
plt.ylabel("Tần suất")
plt.title("Phân phối số lượng phương tiện")

# Biểu đồ xu hướng ùn tắc theo thời gian
plt.subplot(1, 3, 3)
df_grouped = df.groupby("Thời gian")["Mức ùn tắc"].mean()
df_grouped.plot(kind="line", marker="o", color="red")
plt.xlabel("Thời gian")
plt.ylabel("Mức ùn tắc trung bình")
plt.title("Xu hướng mức ùn tắc theo thời gian")

# Hiển thị tất cả biểu đồ
plt.tight_layout()
plt.show()

#------------------------------------
# Xác định đặc trưng (features) và nhãn (label)
X = df[["Số lượng phương tiện"]]  # Biến đầu vào (feature)
y = df["Mức ùn tắc"]  # Biến đầu ra (label)

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (chỉ cần thiết nếu có nhiều đặc trưng)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lr = lr_model.predict(X_test_scaled)

# Khởi tạo và huấn luyện mô hình
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_dt = dt_model.predict(X_test)
# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nĐánh giá mô hình: {model_name}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

# Đánh giá Hồi quy tuyến tính
evaluate_model(y_test, y_pred_lr, "Hồi quy tuyến tính")

# Đánh giá Cây quyết định hồi quy
evaluate_model(y_test, y_pred_dt, "Cây quyết định hồi quy")


