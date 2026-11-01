# Chạy MBA / MBA-A trên Kaggle

## Dán vào một cell và bấm Run, không dùng lệnh

Mở `mba_a_kaggle_cell.py`, sao chép **toàn bộ nội dung** vào một cell Python trên Kaggle. Thêm dữ liệu vào Input, bật GPU trong Notebook Settings, chỉnh dictionary `CONFIG` ở đầu mã rồi bấm Run. Không cần `!python`, terminal, hoặc tải các module Python khác.

Ví dụ thông số cần chú ý trong `CONFIG`:

```python
"data_root": "/kaggle/input",  # Thư mục chứa dataset đã thêm vào Input.
"dataset": "beibei",          # Hoặc "taobao".
"variants": ["B3"],           # Mặc định MBA_A đầy đủ.
"pretrain_epochs": 2,         # Chạy thử trước; huấn luyện thật dùng 100.
"epochs": 2,                  # Chạy thử trước; huấn luyện thật dùng 200.
"output": "/kaggle/working/mba_a_smoke",
```

Khi huấn luyện đầy đủ, tăng lại số epoch và đổi output sang `mba_a_results` để tách kết quả chạy thử. Muốn so sánh MBA với MBA_A, đặt `"variants": ["B0", "B3"]`. Cell chuyển `CONFIG` thành danh sách tham số tường minh, không đọc `sys.argv` nên không gặp lỗi `-f ...kernel.json`. Nếu có nhiều bản CSV trùng tên trong Input, trỏ `data_root` tới đúng thư mục Dataset. Sau khi Run, checkpoint và các CSV so sánh nằm trong thư mục output đã chọn.

## Chạy riêng MBA_A theo tài liệu PDF

**File dùng ngay: `train_mba_a_kaggle.py`.** Đây là script độc lập, mặc định chạy B3 (MBA_A), đã bật trọng số theo người dùng và tăng dần KL. Chỉ tải file này cùng dữ liệu lên Kaggle; không cần `train_mba_kaggle.py` hoặc các module khác.

Trong cell Notebook, tìm script rồi chạy:

```python
from pathlib import Path
scripts = list(Path('/kaggle/input').rglob('train_mba_a_kaggle.py'))
assert len(scripts) == 1, scripts
SCRIPT = str(scripts[0])

# Chạy thử 2 epoch trước khi huấn luyện đầy đủ.
!python -u "{SCRIPT}" --data-root /kaggle/input --dataset beibei \
    --pretrain-epochs 2 --epochs 2 --seeds 2020 \
    --output /kaggle/working/mba_a_smoke
```

Huấn luyện đầy đủ:

```python
!python -u "{SCRIPT}" --data-root /kaggle/input --dataset beibei \
    --seeds 2020 2021 2022 --pretrain-epochs 100 --epochs 200 \
    --patience 20 --alpha 10 --c1 100 --c2 100 --beta 0.8 \
    --warmup-epochs 10 --support-scale 1 --epsilon 0.05 \
    --lambda0 1e-4 --lambda1 1e-6 \
    --output /kaggle/working/mba_a_results
```

Mặc định chỉ huấn luyện MBA_A; thêm `--variants B0 B3` nếu cần so sánh với MBA gốc trong cùng giao thức. Thêm `--variants B0 B1 B2 B3` để thử riêng từng cải tiến. Đổi `--dataset taobao` cho Taobao. Khi có nhiều bản dữ liệu giống tên trong Input, thay `--data-root` bằng thư mục Dataset cụ thể.

Phần còn lại của hướng dẫn về dữ liệu, GPU, checkpoint, validation và kết quả áp dụng cho cả hai script. Script MBA_A dùng cùng engine huấn luyện được đóng gói độc lập; khác mặc định biến thể và thư mục output.

---

Script chính: `train_mba_kaggle.py`. Chỉ cần file này và dữ liệu, không cần chép các module Python khác của dự án.

## 1. Đưa dữ liệu và script lên Kaggle

Tạo một Kaggle Dataset, tải lên script và các CSV sau (có thể giữ nguyên thư mục `data`):

```text
train_mba_kaggle.py
data/
  beibei/
    beibei_buy_train.csv
    beibei_buy_test.csv
    beibei_pv_train.csv
  taobao/
    taobao_buy_train.csv
    taobao_buy_test.csv
    taobao_pv_train.csv
```

Chỉ cần bộ dữ liệu bạn muốn chạy. File dùng dấu tab và có dòng tiêu đề `uid\tsid`, ID số nguyên không âm như dữ liệu hiện tại. `pv_test.csv` không được dùng để huấn luyện. Không cần `.npz` vì bản này dùng MF.

Tạo Notebook, thêm Dataset vào Input, bật GPU trong Settings. Dùng PyTorch, NumPy và pandas đã có trong môi trường. **Không cài requirements.txt cũ của dự án** (`torch==1.10.1`, `scipy==1.7.1`) vào môi trường Kaggle mới. Script không cần SciPy.

Trong một cell Python, xác định đường dẫn:

```python
from pathlib import Path
import torch

scripts = list(Path('/kaggle/input').rglob('train_mba_kaggle.py'))
assert len(scripts) == 1, scripts
SCRIPT = str(scripts[0])
DATA_ROOT = '/kaggle/input'  # Nếu có nhiều bản cùng dataset, trỏ tới thư mục cụ thể.
print(SCRIPT, torch.__version__, 'CUDA:', torch.cuda.is_available())
```

## 2. Kiểm tra dữ liệu

```python
!python "{SCRIPT}" --data-root "{DATA_ROOT}" --dataset beibei --prepare-only
```

Script tự tìm ba CSV; nếu có nhiều bản giống tên, yêu cầu `--data-root` cụ thể. Manifest và split được ghi vào `/kaggle/working/mba_results`.

## 3. Chạy thử ngắn trước

```python
!python -u "{SCRIPT}" --data-root "{DATA_ROOT}" --dataset beibei \
    --variants B0 B3 --seeds 2020 --pretrain-epochs 2 --epochs 2 \
    --output /kaggle/working/mba_smoke
```

Đây chỉ là kiểm tra luồng chạy, không dùng điểm của 2 epoch làm kết luận nghiên cứu. Mặc định tự chọn GPU; thêm `--device cpu` nếu muốn chạy CPU. Nếu CUDA hết bộ nhớ, giảm `--batch-size` và `--eval-batch-size`.

## 4. Huấn luyện và so sánh B0-B3

```python
!python -u "{SCRIPT}" --data-root "{DATA_ROOT}" --dataset beibei \
    --variants B0 B1 B2 B3 --seeds 2020 2021 2022 \
    --pretrain-epochs 100 --epochs 200 --patience 20 \
    --alpha 10 --c1 100 --c2 100 --beta 0.8 \
    --lambda0 1e-4 --lambda1 1e-6 --warmup-epochs 10 \
    --batch-size 2048 --eval-batch-size 128 \
    --output /kaggle/working/mba_results
```

Đổi `--dataset taobao` để chạy Taobao. Cấu hình trên là điểm khởi đầu, chưa được khẳng định tối ưu. Hãy bắt đầu một seed rồi mở rộng số seed theo thời gian GPU còn lại.

| Biến thể | Trọng số theo người dùng | Tăng dần KL |
|---|---|---|
| B0 | Không | Không |
| B1 | Có | Không |
| B2 | Không | Có |
| B3 | Có | Có |

Mỗi seed chỉ huấn luyện hai teacher một lần và dùng chung cho tất cả biến thể. Trọng số theo người dùng được tính bằng n/(n+k), k là trung vị tương tác dương nhân `--support-scale` (mặc định 1); hai trọng số chuẩn hóa có tổng bằng 2 và epsilon mặc định 0.05. Lịch KL tăng tuyến tính đến `--alpha` trong `--warmup-epochs` đầu. Không thay đổi công thức phối hợp dự đoán với beta.

## 5. Kết quả và checkpoint

Trong mỗi thư mục `<dataset>_<fingerprint>`:

```text
data_manifest.json             # Hash dữ liệu, số lượng mẫu và giao thức
splits.npz                     # Train/validation/test để tái sử dụng
teachers/                      # Checkpoint teacher và log validation
B0_seed2020_<config_hash>/
  config.json
  history.jsonl
  best_target.pt
  best_validation.json
  test_per_user.csv
  result.json
...                           # B1/B2/B3 và các seed khác
comparison_per_seed.csv
comparison_mean_std.csv
```

`comparison_mean_std.csv` có Recall/NDCG trung bình, độ lệch chuẩn mẫu và mức tăng tương đối so với B0 trên các seed ghép cặp. Một seed thì std để trống. `result.json` có kết quả theo nhóm người dùng 0, 1-5, 6-10, trên 10 sản phẩm mua train. Thời gian huấn luyện target nằm trong result; tổng thời gian và thời gian tới checkpoint teacher tốt nhất nằm trong metadata teacher. Cộng tổng thời gian hai teacher và target nếu báo cáo chi phí đầy đủ; không coi thời gian nạp cache là thời gian huấn luyện.

```python
import pandas as pd
for path in Path('/kaggle/working/mba_results').rglob('comparison_mean_std.csv'):
    print(path)
    display(pd.read_csv(path))
```

Lưu Notebook Version để giữ output; tải file trong `/kaggle/working` từ phần Output. Hướng dẫn nền tảng: https://www.kaggle.com/docs/notebooks

## 6. Giao thức và phạm vi so sánh

- Đây là bản triển khai **MF** độc lập. Chưa bao gồm LightGCN, đa GPU hoặc resume optimizer giữa epoch.
- B0 giữ công thức BPR pretraining, KL, DN/DP và điều chuẩn theo mã nguồn gốc. Đây **không phải tái lập nguyên xi bảng bài báo**: script chọn checkpoint bằng validation và không gộp pv_test vào train.
- Tách 10% tương tác/người từ buy_train và pv_train; giữ ít nhất một tương tác train. Split seed cố định 42; seed mô hình riêng. ID catalog dùng phạm vi ID của dữ liệu, nhưng các nhãn validation/test không được thêm vào train.
- Mua train được gộp vào nhấp train theo giả định của repo. Mẫu click validation trùng với mua train sẽ được loại khỏi validation nhấp để không đánh giá trên mẫu đã học. Manifest ghi số mẫu này.
- Nếu một cặp đã xuất hiện trong click_train và là mua validation/test thì vẫn có thể làm tín hiệu nhấp, đúng bài toán đa hành vi với click train có sẵn. Không dùng nhãn mua held-out để xây dựng train. CSV không có timestamp nên không tuyên bố đây là dự đoán theo thời gian.
- Đánh giá xếp hạng toàn catalog, loại mua train; giữ cả các sản phẩm chỉ mới nhấp. Validation dùng cùng mặt nạ. Test chỉ chạy sau khi chọn checkpoint bằng validation.
- Teacher mua và nhấp chọn bằng validation tương ứng; target chọn Recall@max(top_k), mặc định Recall@20. Giữ beta cố định cho cả B0-B3 trong mỗi phép so sánh.
- Cần chọn hyperparameter bằng validation. Nếu xem test của nhiều lần chạy để chọn cấu hình thì vẫn gây thiên lệch đánh giá. Không tuyên bố MBA-A tốt hơn trước khi có kết quả nhiều seed.
- Lấy một mẫu âm chưa có trong nhấp train (đã chứa mua train) cho mỗi mẫu target, đúng NSR=1 của thí nghiệm đề xuất. Hai pha DP/DN thay nhau theo batch như repo; dấu của số hạng C2 theo code gốc.

## 7. Chạy lại và tránh nhầm kết quả

Checkpoint teacher có hash dữ liệu, split và cấu hình. Script tái dùng teacher nếu khớp; `--retrain-teachers` buộc huấn luyện lại. Run target hoàn tất được bỏ qua nếu không có `--force`. Run target bị dừng giữa chừng sẽ **chạy lại từ đầu**, không tiếp tục optimizer. Hãy lưu output của phiên Kaggle nếu cần tái dùng sau đó và chép sang thư mục làm việc của phiên mới.

Các CSV tổng hợp chứa những cấu hình/seed được yêu cầu trong lệnh hiện tại. Khi chạy riêng từng biến thể rồi muốn tổng hợp tất cả, chạy lại lệnh với toàn bộ biến thể/seed: các run hoàn tất được nạp lại, không huấn luyện lại. Đổi hyperparameter sẽ tạo thư mục hash mới; không gộp các cấu hình khác nhau thành một bảng trung bình.

## 8. Kiểm thử cục bộ

Chạy tại thư mục dự án bằng Python có torch, numpy, pandas:

```powershell
python -m unittest discover -s kaggle -p "test_*.py" -v
```

Kiểm tra loss và gradient với loss_func.py gốc, lấy mẫu âm, trọng số, mặt nạ đánh giá, Recall/NDCG và luồng train cả bốn biến thể trên dữ liệu nhỏ. Không cần tải file test lên Kaggle để huấn luyện.

## 9. Lỗi Notebook: unrecognized arguments: -f ...kernel.json

Jupyter/Colab tự truyền `-f <kernel.json>` vào chương trình khi chạy mã trực tiếp trong cell. Hai script đã chấp nhận riêng tham số này và vẫn báo lỗi nếu gõ sai tham số huấn luyện. Nếu gặp lỗi, tải lại file `.py` đã cập nhật rồi chạy lại; hoặc thêm dòng sau trước `args = parser.parse_args(argv)` trong bản cũ:

```python
parser.add_argument("-f", dest="_jupyter_kernel_file", help=argparse.SUPPRESS)
```

Cách chạy rõ cấu hình nhất vẫn là `!python "{SCRIPT}" --data-root ...` như các ví dụ bên trên. Nếu muốn gọi từ một cell Python, sau khi đưa file vào thư mục import được:

```python
import train_mba_a_kaggle as mba_a
mba_a.main([
    '--data-root', '/kaggle/input/ten-dataset',
    '--dataset', 'beibei',
    '--output', '/kaggle/working/mba_a_results',
])
```

Danh sách truyền vào `main([...])` chỉ chứa các tham số huấn luyện, không lấy tham số kernel. Trên Colab, đổi đường dẫn dữ liệu thành vị trí đã tải lên trong `/content` và output thành `/content/mba_a_results`. Sửa `-f` không tự động tải dữ liệu hoặc đổi đường dẫn Kaggle sang Colab.
