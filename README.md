# Giải pháp Dự báo Doanh thu - Datathon 2026: The Gridbreakers

## Thành viên nhóm

1. Vương Thành Đạt - Trưởng nhóm (Data Engineering)
2. Trần Nguyên Hưng  (Visualisation)
3. Phạm Phương Thảo  (Modelling)

## Mục tiêu bài toán

Xây dựng mô hình dự báo doanh thu và chi phí (COGS) từ dữ liệu lịch sử, kết hợp:
- Tiền xử lý dữ liệu (đồng bộ, làm sạch)  
- Phân tích khám phá dữ liệu (EDA) và trực quan hóa  
- Tạo đặc trưng (Feature Engineering)  
- Huấn luyện nhiều mô hình dự báo  
- Tối ưu siêu tham số với Optuna  
- Kết hợp (Ensemble) các mô hình để cải thiện độ ổn định và chính xác  

## Cấu trúc thư mục

- `notebooks/`: chứa các notebook phân tích và huấn luyện (R2U_Part1.ipynb, R2U_Part2.ipynb, R2U_Part3.ipynb)  
- `data/`: bộ dữ liệu gốc (ví dụ: `sales.csv`, `promotions.csv`, `web_traffic.csv`, `sample_submission.csv`,...)  
- `submissions.csv`: file submission đầu ra cuối cùng sau khi chạy notebook huấn luyện 
- `requirements.txt`: danh sách các thư viện cần cài đặt  
- `README.md`: tài liệu hướng dẫn (tệp này)  

## Thư viện cần cài đặt

Cài đặt toàn bộ thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

Một số thư viện chính được sử dụng:
- `pandas`, `numpy`: xử lý và phân tích dữ liệu  
- `scikit-learn`: chia tập dữ liệu, đánh giá mô hình, các hàm tiền xử lý (StandardScaler, TimeSeriesSplit, ... )  
- `matplotlib`, `seaborn`: trực quan hóa dữ liệu  
- `statsmodels`: phân tích chuỗi thời gian (kiểm định ADF, phân tích thành phần mùa vụ, ACF/PACF)  
- `lightgbm`, `xgboost`, `catboost`: các mô hình dự báo chính (gradient boosting)  
- `optuna`: tối ưu hoá siêu tham số tự động  
- `shap`: giải thích mô hình (tính giá trị SHAP)  
- `joblib`, `tqdm`: tiện ích lưu nạp mô hình và hiển thị tiến độ  
- `openpyxl`: đọc/ghi file Excel (nếu cần sử dụng)  

## Quy trình thực hiện

### 1. Khám phá và tiền xử lý dữ liệu

Trong notebook **R2U_Part2.ipynb** (Phần 2: Data Engineering), chúng tôi thực hiện:
- Đọc các bảng dữ liệu (sales, traffic, promotions, sample_submission, ...) và kiểm tra kích thước, phạm vi ngày (Sales train: 04/2012–12/2022, Test: 01/2023–06/2024).  
- Đồng bộ định dạng cột, ép kiểu dữ liệu (chuyển cột `Date` sang kiểu datetime, đảm bảo mã sản phẩm, mã khách hàng là chuỗi, ...).  
- Kiểm tra và xử lý giá trị thiếu (nếu có) và loại bỏ ngoại lai (outliers) trong dữ liệu bán hàng.  
- Gom nhóm dữ liệu (aggregation) và tổng hợp thành “Single Source of Truth” theo ngày: tính tổng doanh thu (`Revenue`) và giá vốn (`COGS`) từ các bảng nguồn.  
- Các bước chuẩn bị khác: điền giá trị thiếu cho traffic, ghép dữ liệu khuyến mãi để xác định thời gian khuyến mãi, v.v.  

### 2. Phân tích dữ liệu và trực quan hóa

Trong notebook **R2U_Part2.ipynb** (Phần 3: Phân tích dữ liệu), chúng tôi thực hiện:
- Phân tích cấu trúc chuỗi thời gian: phân rã thành phần (decompose) để xác định xu hướng và mùa vụ; kiểm định ADF để kiểm tra tính dừng của chuỗi; vẽ ACF/PACF để hiểu độ tự tương quan.  
- Phân tích các yếu tố ảnh hưởng đến doanh thu qua nhiều khía cạnh (ví dụ: hiệu quả doanh thu theo sản phẩm/danh mục, tác động của khuyến mãi, hành vi khách hàng theo nhóm/quốc gia, hiệu quả vận hành kho, v.v.).  
- Sử dụng trực quan hóa (biểu đồ đường, biểu đồ cột, heatmap, biểu đồ Pareto, v.v.) để rút ra insight: xác định rõ chu kỳ mùa vụ (sales tăng/giảm theo tháng/quý), ảnh hưởng của các ngày lễ/ngày trả lương/khuyến mãi, mối quan hệ giữa traffic và doanh thu, v.v.  
- Rút ra các đặc trưng quan trọng để đưa vào mô hình: tạo hơn 60 biến mới thuộc các nhóm như **lag features** (doanh thu các ngày trước đó), **rolling statistics** (trung bình và thống kê trượt của Revenue/COGS), đặc trưng **lịch** (tháng, ngày trong tuần, tuần trong năm, ngày lễ, ngày trả lương, double-day,…), **Fourier** cho thành phần mùa vụ phi tuyến, **xu hướng chung** và **tín hiệu ngoại sinh** (traffic, khuyến mãi, v.v.).  

### 3. Huấn luyện mô hình

Trong notebook **R2U_Part3.ipynb** (tiếp tục R2U_Part2), quy trình huấn luyện mô hình bao gồm:
- Chia dữ liệu theo thời gian (time-based split): Training dùng dữ liệu đến hết năm 2021, Validation dùng năm 2022, Test dự báo năm 2023-2024.  
- Thiết lập mô hình baseline đơn giản: Naive Forecast (giả định doanh thu ngày mai = ngày trước), Seasonal Naive (dùng giá trị cùng ngày tuần trước), và trung bình trượt, để làm benchmark RMSE ban đầu.  
- Huấn luyện các mô hình chính: **LightGBM**, **XGBoost**, và **CatBoost**. Sử dụng kỹ thuật seed averaging để tăng tính ổn định (chạy lặp với nhiều giá trị `random_state` khác nhau và lấy trung bình dự báo).  
- Tối ưu siêu tham số: Sử dụng **Optuna** (khoảng 50 trials) để tìm bộ siêu tham số tốt nhất cho LightGBM và XGBoost. CatBoost sử dụng thiết lập mặc định kết hợp **early stopping**.  
- Kết hợp mô hình (Ensemble): Tạo dự báo cuối cùng bằng cách kết hợp có trọng số các dự báo của ba mô hình (ví dụ: 40% LightGBM, 20% XGBoost, 40% CatBoost), giúp giảm sai số so với mô hình đơn lẻ.  
- Đánh giá mô hình: Sử dụng các chỉ số MAE, RMSE, R² trên tập Validation (năm 2022) để đánh giá hiệu quả.  
- Dự báo đa bước (multi-step): Thực hiện dự báo tuần tự 548 ngày (năm 2023-2024) bằng cách phương pháp recursive forecasting (mỗi bước dự báo sẽ dùng kết quả bước trước, tránh leak thông tin tương lai).  
- Sau khi dự báo được cột doanh thu, tính toán cột **COGS** tương ứng: sử dụng tỷ lệ biên lợi nhuận trung bình (COGS/Revenue) theo từng tháng từ dữ liệu lịch sử, áp dụng cho giá trị dự báo và đảm bảo ràng buộc kinh doanh `COGS < Revenue`.  
- Kết quả đầu ra: file `submission.csv` gồm các cột `Date`, `Revenue`, `COGS`.  

### 4. Tổng kết quy trình

Các bước chính đã thực hiện:
- **EDA & Kiểm định**: Phân tích chuỗi thời gian (decompose, ACF/PACF) xác nhận xu hướng tăng chung và mùa vụ rõ; kiểm định ADF cho thấy chuỗi cần được xử lý phi-stationary.  
- **Feature Engineering**: Tạo ra hơn 60 biến mới từ 7 nhóm đặc trưng như đã mô tả ở trên (lag, rolling, lịch, sự kiện, Fourier, xu hướng, ngoại sinh).  
- **Mô hình baseline**: Thiết lập benchmark từ các mô hình đơn giản (Naive, Seasonal, Rolling) để đánh giá bước tiếp theo.  
- **Huấn luyện và tinh chỉnh**: Huấn luyện LightGBM, XGBoost, CatBoost; sử dụng Optuna để tìm siêu tham số tốt; ứng dụng seed averaging và ensemble.  
- **Dự báo & Xuất file**: Thực hiện dự báo tuần tự cho tập test (năm 2023-2024), áp dụng ràng buộc doanh thu & COGS, và sinh file `submission.csv`.  

## Hướng dẫn chạy lại kết quả

### Bước 1: Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy notebook huấn luyện

- Chạy lần lượt các notebook theo thứ tự:  
  1. `notebooks/R2U_Part1.ipynb` (các câu hỏi kiểm thử ban đầu)  
  2. `notebooks/R2U_Part2.ipynb` (tiền xử lý và phân tích dữ liệu)  
  3. `notebooks/R2U_Part3.ipynb` (huấn luyện mô hình, tinh chỉnh tham số, dự báo)  
- Hoặc mở và chạy trực tiếp notebook `R2U_Part3.ipynb`, vì trong đó đã tích hợp quy trình huấn luyện và tạo file submission.  

### Bước 3: Tạo file dự báo

Sau khi chạy xong các bước huấn luyện và dự báo, file `submission.csv` sẽ được lưu trong thư mục `submissions/`. Đây là file cần nộp lên hệ thống chấm điểm (bao gồm cột `Date`, `Revenue`, `COGS`).  

## Ghi chú quan trọng

- Đảm bảo dữ liệu gốc (`sales.csv`, `promotions.csv`, `web_traffic.csv`, `sample_submission.csv`, ...) được đặt trong thư mục `data/` với định dạng đúng.  
- Sử dụng **Python 3.x** và đã cài đặt đầy đủ các thư viện trong `requirements.txt`.  
- Các notebook đã thiết lập seed ngẫu nhiên (`random_state`) để kết quả có thể tái lập.  
- Mô hình đảm bảo thỏa mãn ràng buộc kinh doanh: luôn giữ `COGS < Revenue`.  

## Liên hệ

- Vương Thành Đạt - Trưởng nhóm  
   Email: vuongdat0306@gmail.com
