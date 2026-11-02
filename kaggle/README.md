# Huấn luyện MBA trên Kaggle

**Bắt đầu:** import `MBA_Taobao_Beibei_Kaggle.ipynb`, thêm Kaggle Dataset chứa các CSV, bật GPU, Run All. Notebook tự chứa mã nguồn, không cần clone repository hoặc bật Internet.

Các gói dựng sẵn:

- `output/kaggle/mba_kaggle_code.zip`: notebook, source, kiểm thử, phân tích, LICENSE.
- `output/kaggle/mba_taobao_beibei_data.zip`: tám CSV từ `data/`, không kèm adjacency cache. Upload ZIP thành Kaggle Dataset và kiểm tra CSV đã được giải nén trước khi Add Input.

Mặc định chạy **Taobao + Beibei**, **MBA đối chứng + adaptive**, seed 2020. Đặt `SEEDS=[2020,2021,2022]` để báo cáo nhiều seed; `RUN_ABLATIONS=True` để chạy các biến thể bỏ từng thành phần. Dùng `SMOKE_ONLY=True` chỉ để kiểm tra vận hành; kết quả debug không tham gia bảng nghiên cứu. Tắt smoke và đổi `RUN_ROOT` trước khi huấn luyện đầy đủ.

Hai variant:

| Variant | Nội dung |
|---|---|
| `mba` | Hai teacher BPR đóng băng, bốn denoiser MF, KL và DP/DN theo source; sampler và validation đã sửa |
| `adaptive` | Thêm trọng số KL theo user/agreement, warmup, warm-start, BPR mua hàng, negative pool, cap click-only, L2 denoiser và trộn dự đoán theo user |

Đọc `docs/phan_tich_cai_tien_MBA.md` trước khi diễn giải kết quả. Bản `mba` dùng giao thức validation khác bài gốc; bản adaptive chưa có bằng chứng tăng chất lượng từ huấn luyện đầy đủ. Không so trực tiếp số mới với Table 2 để kết luận cải tiến.

## Chạy từ script

```bash
python kaggle/train_mba.py --help
python kaggle/train_mba.py --data-root data --dataset taobao --variant mba --output output/experiment
python kaggle/train_mba.py --data-root data --dataset taobao --variant adaptive --output output/experiment
python kaggle/train_mba.py --data-root data --dataset beibei --variant adaptive --output output/experiment
python kaggle/summarize_results.py --root output/experiment
```

Dùng `--teacher-cache output/teacher_cache` cho các run chia sẻ cùng cấu hình pretrain. Cache chỉ được dùng lại nếu chữ ký dữ liệu, split, cấu hình và môi trường khớp. Đầu ra có sẵn không bị ghi đè; hãy dùng `--output` mới. Checkpoint lưu để suy luận, chưa hỗ trợ resume đầy đủ optimizer/RNG.

Giữ `--split-seed 2026` giữa các phương pháp và seed. `--seed` thay đổi khởi tạo/lấy mẫu, không thay đổi validation split. Có thể dùng `--backbone lightgcn`; MF là lựa chọn mặc định dựa trên kết quả bài báo. Mỗi optimizer step LightGCN vẫn phải lan truyền toàn graph. Một GPU được dùng, không tự dùng GPU thứ hai.

Ví dụ ablation:

```bash
python kaggle/train_mba.py --data-root data --dataset taobao --variant adaptive --run-label without_adaptive_kl --no-adaptive-kl --output output/experiment
python kaggle/train_mba.py --data-root data --dataset taobao --variant adaptive --run-label without_bpr --bpr-weight 0 --output output/experiment
```

## Dữ liệu và giao thức

- Cần `{dataset}_buy_train.csv`, `{dataset}_buy_test.csv`, `{dataset}_pv_train.csv` trong cùng thư mục. `pv_test` được bỏ qua có chủ đích.
- Tự tìm dưới `--data-root`, hỗ trợ TSV có header như repo, CSV số có/không header, và reindex ID dùng chung.
- Holdout validation theo user từ train, giữ tối thiểu một cặp fit. Mục tiêu 10%, nhưng min-one và làm tròn khiến tỷ lệ thực tế lớn hơn, nhất là Taobao.
- Không thêm purchase validation/test vào graph. Click đã có sẵn có thể trùng purchase giữ lại; không có timestamp để chứng minh thứ tự thời gian.
- Chọn checkpoint theo validation Recall@20. Test full catalog, che toàn purchase-train, không che mọi click.
- Ghi checksum, cấu hình, môi trường, mapping, splits, loss, metric từng user và checkpoint. Kết quả tổng hợp bỏ qua mọi `debug=true`.
- `mean_std.csv` báo mean/std giữa seed. `paired_differences.csv` có CI bootstrap theo user điều kiện trên các model seed, không thay thế kiểm định giữa các training run độc lập.

## Kiểm tra và dựng lại notebook

```bash
python -m unittest discover -s kaggle -p test_train_mba.py -v
python kaggle/audit_data.py --data-root data --output docs/data_audit.json
python kaggle/build_notebook.py --include-data
```

Phụ thuộc runtime: Python >=3.9, torch >=1.10, NumPy, pandas. Dùng thư viện có sẵn của Kaggle; không cài `requirements.txt` gốc vào notebook mới. Đã kiểm thử CPU và smoke hai dataset thật; chưa chạy huấn luyện đầy đủ trên Kaggle GPU.

Nguồn: [bài báo MBA](https://arxiv.org/abs/2305.05585v1), [source tác giả, MIT](https://github.com/LiuXiangYuan/MBA). Giữ LICENSE khi chia sẻ code.
