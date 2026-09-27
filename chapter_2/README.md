# Hướng dẫn chạy code — Chapter 2: Graph Embeddings
File hướng dẫn này chỉ áp dụng cho **Chapter 2 — Graph Embeddings** (mục 2.1 Node2Vec, 2.2 GNN Embeddings, 2.3 Semi-supervised Classification). Các chapter khác của sách không nằm trong phạm vi hướng dẫn này.
### 📂 [Click vào đây để mở trực tiếp Chapter 2](https://github.com/dxhai17/GNN_AI_C-3/tree/master/chapter_2)

### [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dxhai17/GNN_AI_C-3/blob/master/chapter_2/archived_files/Chapter_2_3_GNN_Embeddings.ipynb)
---
## Chapter 2.3: GNN Embeddings
## 1. Thông tin repo

- **Repo gốc (chính thức của sách):** https://github.com/keitabroadwater/gnns_in_action
- **Repo fork của nhóm (dùng để clone):** https://github.com/dxhai17/GNN_AI_C-3.git

> ⚠️ **Lưu ý:** Repo fork có thể đặt tên thư mục/file khác đôi chút so với repo gốc. Trước khi chạy, hãy kiểm tra lại đúng tên thư mục Chapter 2 trong repo fork (thường là `chapter_2/`, `ch2/`, hoặc tên notebook dạng `Chapter_2_3_GNN_Embeddings.ipynb`) và sửa lại đường dẫn `cd` bên dưới cho khớp.

---

## 2. Clone repo về máy

```bash
git clone https://github.com/dxhai17/GNN_AI_C-3.git
cd GNN_AI_C-3
```

Sau khi clone, tìm đến thư mục/notebook tương ứng Chapter 2 (ví dụ `Chapter_2_3_GNN_Embeddings.ipynb`).

---

## 3. Cài đặt môi trường

Khuyến nghị **dùng môi trường ảo (virtual environment) riêng** cho từng chương, tránh xung đột thư viện với các dự án khác trên máy.

### 3.1. Trên Ubuntu / Linux (khuyến nghị)

```bash
# Kiểm tra phiên bản Python (khuyến nghị Python 3.10 – 3.12)
python3 --version

# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip

# Cài các thư viện cần thiết cho Chapter 2
pip install torch torch_geometric
pip install pandas networkx tqdm scikit-learn matplotlib umap-learn node2vec
```

### 3.2. Trên Windows

```powershell
# Kiểm tra phiên bản Python
python --version

# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# Nâng cấp pip
pip install --upgrade pip

# Cài các thư viện cần thiết
pip install torch torch_geometric
pip install pandas networkx tqdm scikit-learn matplotlib umap-learn node2vec
```

### 3.3. Cài đặt Jupyter Notebook / VSCode

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=gnn_ch2 --display-name "GNN Chapter 2"
```

Trong VSCode: mở file `.ipynb` → chọn kernel **"GNN Chapter 2"** (góc trên bên phải notebook) trước khi chạy bất kỳ cell nào.

---

## 4. Thứ tự chạy notebook

1. Chạy tuần tự từ cell đầu tiên — **không bỏ qua cell**, vì các biến (`gml_graph`, `embeddings`, `data`, `model`...) được định nghĩa và sử dụng lại xuyên suốt các mục 2.1 → 2.2 → 2.3.
2. Nếu notebook yêu cầu file dữ liệu (`.gml` của Political Books dataset), đảm bảo đường dẫn file trong lệnh đọc dữ liệu (ví dụ `polbooks_gml(...)`) trỏ đúng đến vị trí file đã tải kèm trong repo.
3. Với các cell dùng `EmbeddingDataset` (mục 2.3), lần chạy **đầu tiên** có thể mất thời gian lâu hơn vì phải xử lý (`process()`) dữ liệu thô thành tensor — các lần chạy sau sẽ nhanh hơn nhờ dữ liệu đã được cache lại.

---

## 5. Troubleshooting — Các lỗi thường gặp và cách xử lý

Đây là các lỗi thực tế nhóm đã gặp khi chạy notebook Chapter 2 trên các máy khác nhau (Windows và Ubuntu), kèm nguyên nhân và cách sửa.

### 5.1. `ModuleNotFoundError: No module named 'torch'` (hoặc `pandas`, `sklearn`,...)

**Nguyên nhân:** Clone code từ GitHub chỉ tải về mã nguồn, **không tự động cài thư viện phụ thuộc (dependencies)**.

**Cách sửa:**
```bash
pip install torch pandas networkx tqdm torch_geometric scikit-learn matplotlib umap-learn node2vec
```

Nếu vẫn báo thiếu module sau khi cài, kiểm tra xem có đang chạy đúng môi trường (kernel) đã cài thư viện hay không:
```python
import sys
print(sys.executable)
```
Đảm bảo đường dẫn này trỏ đúng vào `.venv` đã tạo ở Bước 3.

---

### 5.2. Windows: `DLL load failed while importing ccalendar: An Application Control policy has blocked this file`

**Nguyên nhân:** Windows Defender Application Control (WDAC) hoặc Smart App Control chặn file `.pyd` (biên dịch sẵn) của thư viện `pandas` vì không đạt yêu cầu "Enterprise signing level".

**Cách sửa (khuyến nghị — không cần gỡ chính sách bảo mật):**
- **Cách nhanh nhất:** Chuyển hẳn sang chạy trên **Ubuntu/WSL** (như nhóm đã áp dụng) — tránh hoàn toàn giới hạn này của Windows.
- **Nếu bắt buộc dùng Windows:** kiểm tra xem có phải Smart App Control (Windows 11) đang bật không:
  ```
  Windows Security → App & browser control → Smart App Control settings → Off
  ```
---

### 5.3. `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

**Nguyên nhân:** Từ PyTorch 2.2 trở đi, tham số `verbose` trong các learning rate scheduler (bao gồm `ReduceLROnPlateau`) đã bị **deprecated và loại bỏ**. Code gốc trong repo sách được viết ở thời điểm PyTorch còn hỗ trợ tham số này.

**Cách sửa — bỏ tham số `verbose`:**
```python
# Trước (lỗi trên PyTorch mới):
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Sau (đúng với PyTorch mới):
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
```


---

### 5.4. `UnpicklingError` khi load dataset: `Unsupported global: GLOBAL torch_geometric.data.data.DataEdgeAttr`

**Nguyên nhân:** Từ **PyTorch 2.6**, tham số `weights_only` trong `torch.load()` mặc định đổi từ `False` sang `True` (nhằm tăng bảo mật, chỉ cho phép load tensor thuần, không cho load custom class như `DataEdgeAttr` của `torch_geometric`).

**Cách sửa** — tìm dòng `torch.load(...)` trong class dataset (thường trong class kế thừa `InMemoryDataset`, ví dụ `EmbeddingDataset`) và thêm `weights_only=False`:

```python
# Trước (lỗi trên PyTorch >= 2.6):
self.data, self.slices = torch.load(self.processed_paths[0])

# Sau (đúng với PyTorch mới):
self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
```

> ⚠️ Chỉ dùng `weights_only=False` khi **chắc chắn file `.pt` đến từ nguồn đáng tin cậy** (ví dụ: do chính notebook này tự tạo ra ở bước xử lý trước đó, hoặc từ repo chính thức của sách) — không áp dụng cách này với file `.pt` tải từ nguồn lạ, vì có nguy cơ thực thi mã độc.

**Sau khi sửa:** chạy lại cell định nghĩa class dataset trước, rồi mới chạy lại cell khởi tạo `dataset = EmbeddingDataset(...)`.

---

## 6. Nguyên nhân chung của các lỗi trên

Toàn bộ các lỗi ở phần Troubleshooting đều xuất phát từ **một nguyên nhân gốc chung**: code trong repo sách được viết tại một thời điểm cụ thể với các phiên bản thư viện (PyTorch, PyTorch Geometric...) tại lúc đó, nhưng các thư viện này liên tục cập nhật theo thời gian và đôi khi **loại bỏ hoặc thay đổi API cũ (breaking changes)** mà không giữ tương thích ngược.

Nếu trong quá trình chạy các mục khác của notebook (2.1, 2.2, 2.3) tiếp tục gặp lỗi dạng `TypeError: ... got an unexpected keyword argument ...` hoặc `AttributeError`, nhiều khả năng cũng cùng nguyên nhân này — có thể tra cứu nhanh bằng cách tìm tên hàm/tham số gây lỗi kèm từ khóa "deprecated" trên tài liệu chính thức của PyTorch/PyTorch Geometric để tìm tham số thay thế tương ứng.

---