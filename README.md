# 📈 Quantitative Trading Strategy Backtester: SMA + OBV

Ứng dụng web định lượng (Quantitative Finance Web App) xây dựng trên nền tảng **Streamlit**, dùng để kiểm định (backtest), tối ưu hóa tham số (**Hyperopt**) và trực quan hóa hiệu quả của chiến lược giao dịch kết hợp giữa **Đường trung bình động (SMA)** và **Khối lượng cân bằng (On-Balance Volume - OBV)** từ dữ liệu lịch sử cổ phiếu (mẫu: ACB giai đoạn 2014 – 2023).

---

## 🌟 Tính Năng Nổi Bật

1. **Chuẩn hóa dữ liệu linh hoạt**:
   - Tự động nhận diện dữ liệu mẫu `ACB.csv` (2014 - 2023) hoặc cho phép tải lên file CSV bất kỳ (hỗ trợ các cột Date, Open, High, Low, Close, Volume).
2. **Quy trình phân chia Train / Test chuẩn mực (In-sample & Out-of-sample)**:
   - Tập **TRAIN** (2014-01-01 đến 2019-12-31): Dùng để nghiên cứu, kiểm định và chạy thuật toán tối ưu hóa siêu tham số Hyperopt.
   - Tập **TEST** (2020-01-01 đến 2023-12-31): Dùng để kiểm tra độc lập (Out-of-sample) với bộ tham số cố định từ Train, tránh hiện tượng quá khớp (Overfitting / Look-ahead bias).
3. **Đa dạng hóa biến thể chiến lược**:
   - **SMA Crossover**: Giao cắt giữa đường trung bình nhanh (`ma_short`) và đường trung bình chậm (`ma_long`).
   - **OBV Crossover**: Giao cắt giữa chỉ báo dòng tiền tích lũy OBV và đường trung bình động của chính nó (`obv_window`).
   - **Kết hợp AND**: Mua khi cả hai cùng báo Mua, Bán khi cả hai cùng báo Bán (hỗ trợ thêm chế độ *Regime Filter* nâng cao).
   - **Kết hợp OR**: Mua khi một trong hai báo Mua, Bán khi một trong hai báo Bán.
   - **Benchmark**: So sánh trực tiếp với chiến lược Mua và Nắm giữ thụ động (Buy & Hold).
4. **Tích hợp thuật toán tối ưu hóa Hyperopt (TPE Algorithm)**:
   - Tự động dò tìm bộ tham số tối ưu trên tập Train theo 4 mục tiêu: Tối đa hóa Lợi nhuận (Return %), Tối đa hóa Sharpe Ratio, Tối thiểu hóa Max Drawdown hoặc Lệnh thua lớn nhất.
5. **Trực quan hóa tương tác cao cấp (Plotly Interactive Charts)**:
   - Biểu đồ tăng trưởng vốn (Equity Curve) đa chiến lược kèm biểu đồ sụt giảm tài khoản (Underwater Drawdown %).
   - Biểu đồ nến kỹ thuật (Candlestick) tích hợp các đường SMA, điểm Mua/Bán (Buy/Sell markers), khối lượng giao dịch và chỉ báo OBV.
   - Nhật ký giao dịch chi tiết từng lệnh (Entry/Exit Date, Price, PnL %, Duration) kèm nút tải file CSV.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                  # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt        # Danh sách thư viện phụ thuộc để triển khai
├── README.md               # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv                 # File dữ liệu mẫu cổ phiếu ACB (chuẩn hóa)
├── ACB (2).csv             # File dữ liệu gốc đính kèm
└── SMA_+_OBV (1).ipynb     # Jupyter Notebook nghiên cứu gốc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Đã cài đặt **Python 3.9 - 3.11** trên máy tính.
- Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`).

### 2. Các bước khởi chạy

**Bước 1: Mở Terminal / PowerShell tại thư mục dự án:**
```bash
cd "đường_dẫn_tới_thư_mục_chứa_code"
```

**Bước 2: Tạo và kích hoạt môi trường ảo (khuyến nghị):**
- Trên Windows:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- Trên macOS / Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**Bước 3: Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

**Bước 4: Khởi động ứng dụng Streamlit:**
```bash
streamlit run app.py
```
Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Tải Lên GitHub & Deploy Miễn Phí Trên Streamlit Community Cloud

### Bước 1: Đẩy mã nguồn lên GitHub

1. Tạo một repository mới trên [GitHub](https://github.com/new) (ví dụ đặt tên: `sma-obv-quant-strategy`).
2. Mở Terminal tại thư mục dự án và chạy các lệnh Git sau:
   ```bash
   git init
   git add app.py requirements.txt README.md "ACB.csv"
   git commit -m "feat: initial commit for SMA + OBV Quant App"
   git branch -M main
   git remote add origin https://github.com/<tên_tài_khoản_github>/sma-obv-quant-strategy.git
   git push -u origin main
   ```

### Bước 2: Deploy lên Streamlit Community Cloud

1. Truy cập vào [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền thông tin ứng dụng:
   - **Repository**: Chọn repo vừa tạo (`<tên_tài_khoản_github>/sma-obv-quant-strategy`).
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
4. Nhấn **"Deploy!"**.
5. Trong vòng 1 - 2 phút, Streamlit Cloud sẽ tự động cài đặt các dependencies từ `requirements.txt` và cung cấp đường link web công khai (ví dụ: `https://your-app-name.streamlit.app`) để bạn chia sẻ cho bạn bè, đồng nghiệp hoặc giảng viên đánh giá!

---

## 📊 Tóm Tắt Kết Quả Nghiên Cứu Định Lượng

Dưới đây là kết quả kiểm định thực nghiệm trên dữ liệu cổ phiếu ACB với bộ tham số tối ưu từ Train (`ma_short=90`, `ma_long=240`, `obv_window=35`):

| Chiến Lược | Lợi Nhuận TRAIN (2014-2019) | Sharpe TRAIN | Lợi Nhuận TEST (2020-2023) | Sharpe TEST | Max DD TEST | Số Lệnh TEST |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SMA Crossover (90, 240)** | +196.41% | 0.94 | +17.24% | 0.58 | -13.51% | 0 (giữ vị thế) |
| **OBV Crossover (35)** | +248.51% | 1.13 | +53.63% | 0.54 | -53.14% | 43 |
| **SMA + OBV (AND)** | 0.00% | 0.00 | 0.00% | 0.00 | 0.00% | 0 |
| **⭐ SMA + OBV (OR)** | **+288.26%** | **1.23** | **+72.58%** | **0.66** | **-47.41%** | **43** |
| **Buy & Hold (ACB)** | - | - | +48.91% | - | - | - |

### 💡 Nhận Định Cốt Lõi:
1. **Tại sao điều kiện AND cho 0 lệnh?**
   - Tín hiệu giao cắt SMA chu kỳ dài (90/240) diễn ra rất hiếm (vài năm 1 lần), trong khi OBV (35) biến động liên tục theo phiên. Xác suất để cả 2 sự kiện giao cắt cùng xảy ra **chính xác vào cùng một ngày** gần như bằng 0.
   - *Ứng dụng cung cấp tùy chọn "Regime Filter" để giải quyết vấn đề này.*
2. **Tại sao điều kiện OR đạt hiệu quả cao nhất?**
   - Sự kết hợp **OR** tận dụng được độ nhạy sớm của dòng tiền OBV để vào lệnh ở vùng giá chân sóng, đồng thời tận dụng bộ lọc xu hướng của SMA để không bỏ lỡ các đợt tăng giá lớn, mang lại lợi nhuận cao nhất cả trên Train (+288.26%) và Test (+72.58%).

---

## 🛠️ Công Nghệ Sử Dụng

- **Frontend & Framework**: [Streamlit](https://streamlit.io/)
- **Xử lý số liệu**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Phân tích kỹ thuật**: [ta (Technical Analysis Library in Python)](https://github.com/bukosabino/ta)
- **Engine kiểm thử**: [Backtesting.py](https://kernc.github.io/backtesting.py/) & Custom Quant Simulator
- **Tối ưu hóa siêu tham số**: [Hyperopt](http://hyperopt.github.io/hyperopt/) (Tree of Parzen Estimators - TPE)
- **Đồ thị tương tác**: [Plotly Graph Objects](https://plotly.com/python/)

---

## 📜 Bản Quyền & Miễn Trừ Trách Nhiệm
- Ứng dụng được xây dựng phục vụ mục đích nghiên cứu học thuật và kiểm định chiến lược định lượng.
- Các kết quả backtest trong quá khứ không đảm bảo cho tỷ suất sinh lời trong tương lai. Nhà đầu tư cần cân nhắc các yếu tố trượt giá, thanh khoản và chu kỳ thanh toán T+2.5 khi áp dụng trên thị trường thực tế.
