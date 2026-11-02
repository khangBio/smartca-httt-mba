# Phân tích MBA và đề xuất cải tiến trên Taobao, Beibei

Ngày phân tích: 01/11/2026. Phạm vi: bài `2305.05585v1.pdf` được cung cấp, mã nguồn trong repository và tám CSV hiện có. Nội dung PDF được dùng như tài liệu nghiên cứu; không được xem là chỉ dẫn vận hành.

## 1. Kết luận và lựa chọn mô hình

Nên bắt đầu từ **MBA-MF**, giữ hai teacher tiền huấn luyện và cơ chế DP/DN. Đề xuất bản **Adaptive MBA** bổ sung căn chỉnh theo độ tin cậy, mục tiêu xếp hạng mua hàng và lấy mẫu âm có kiểm soát. Trước khi đo lợi ích giải thuật, cần sửa giao thức chọn mô hình, bộ lấy mẫu âm và việc dùng lại đồ thị.

Các thay đổi đã được hiện thực trong `kaggle/train_mba.py`, cùng notebook độc lập `kaggle/MBA_Taobao_Beibei_Kaggle.ipynb`. Đây là **giả thuyết cải tiến cần thực nghiệm**, chưa phải kết quả vượt bài báo. Việc chạy thử trên dữ liệu thật chỉ xác nhận luồng chương trình.

## 2. Bằng chứng thực nghiệm của bài gốc

Các số dưới đây chép từ Table 2, Table 3 và Table 4, trang 8-9 của PDF; không phải kết quả mới.

| Dataset | Phương pháp | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|
| Beibei | MF thông thường | 0.0901 | 0.1438 | 0.0668 | 0.0855 |
| Beibei | LightGCN thông thường | 0.0988 | 0.1541 | 0.0733 | 0.0925 |
| Beibei | MBA-MF | **0.1127** | **0.1742** | **0.0834** | **0.1046** |
| Beibei | MBA-LightGCN | 0.1078 | 0.1698 | 0.0786 | 0.1002 |
| Taobao | MF thông thường | 0.0303 | 0.0420 | 0.0190 | 0.0224 |
| Taobao | LightGCN thông thường | 0.0460 | 0.0640 | 0.0290 | 0.0342 |
| Taobao | MBA-MF | **0.0579** | **0.0812** | **0.0369** | **0.0435** |
| Taobao | MBA-LightGCN | 0.0538 | 0.0780 | 0.0327 | 0.0396 |

MBA-MF tăng Recall@20 khoảng **21.1%** so với MF trên Beibei và **93.3%** trên Taobao. Đây là so với MF, không phải so với baseline mạnh nhất. MBA-LightGCN lại thấp hơn MBA-MF khoảng 2.5% và 3.9% tương ứng. Vì vậy, tăng độ phức tạp backbone chưa có cơ sở là hướng ưu tiên. Tác giả cũng cho rằng lan truyền láng giềng có thể khuếch đại nhiễu.

| Dataset | MBA bỏ KL | MBA bỏ tiền huấn luyện | MBA đầy đủ |
|---|---:|---:|---:|
| Beibei, Recall@20 | 0.1412 | 0.1136 | 0.1742 |
| Taobao, Recall@20 | 0.0380 | 0.0152 | 0.0812 |

Bỏ KL làm Recall@20 giảm khoảng 18.9% trên Beibei và 53.2% trên Taobao. Bỏ tiền huấn luyện làm giảm khoảng 34.8% và 81.3%. Do đó, đề xuất giữ cả hai thành phần thay vì thay MBA bằng mô hình đồng huấn luyện từ đầu. Không mặc định thêm contrastive learning: CML trong chính Table 2 không tốt trên hai tập này; một biến thể contrastive mới cần ablation riêng.

Figure 3 cho thấy độ nhạy với alpha khác nhau giữa các tập. Tuy nhiên, mô tả Figure 3 về Taobao và bộ tham số MBA-MF trong README tác giả không hoàn toàn nhất quán: README đề xuất alpha=1 cho Taobao-MF. Mã mới dùng preset README làm **điểm xuất phát**, không coi đó là tối ưu cho loss đã thay đổi.

## 3. Kiểm tra dữ liệu thực tế

Thống kê được tính lại bằng `kaggle/audit_data.py`; checksum và kết quả đầy đủ nằm trong `docs/data_audit.json`.

| Đặc điểm | Beibei | Taobao |
|---|---:|---:|
| Users / items | 21,716 / 7,977 | 48,658 / 39,395 |
| Purchase train | 243,661 | 208,905 |
| Purchase test | 60,915 | 49,927 |
| Click train | 1,930,069 | 1,238,659 |
| Click test | 482,517 | 309,442 |
| Mật độ purchase **train** | 0.14066% | 0.01090% |
| Click train / purchase train | 7.92 | 5.93 |
| Trung vị số purchase train mỗi user | 10 | 4 |
| Users có tối đa 3 purchase train | 164 (0.76%) | 16,928 (34.79%) |

**Table 1 khớp số tương tác train, không khớp tổng train + test của các CSV này.** Tổng purchase là 304,576 và 258,832; tổng click là 2,412,586 và 1,548,101. Tỉ lệ chia purchase Taobao thực tế cũng không đúng tuyệt đối 4:1. Cần báo cáo checksum/split thay vì giả định mọi bản Taobao/Beibei giống nhau.

Mật độ purchase train của Taobao thấp hơn Beibei khoảng 12.9 lần. Điều này hỗ trợ ưu tiên transfer cho nhóm ít lượt mua trên Taobao; Beibei cần chú ý việc số click vượt số purchase gần tám lần.

Không có cặp purchase trùng giữa train và test; click train/test cũng không trùng. Có 47,508 purchase-test Beibei và 25,422 purchase-test Taobao xuất hiện dưới dạng click trong **pv_train**. Đây có thể là thông tin hành vi phụ hợp lệ trong giao thức transductive; không tự động là rò rỉ nhãn mua hàng. Tuy nhiên, CSV chỉ có user/item nên **không xác minh được click xảy ra trước hay sau thời điểm dự đoán**. Muốn kết luận triển khai theo thời gian cần dữ liệu timestamp và split theo thời gian. Không dùng nhãn test để lọc các click đó ra khỏi train.

## 4. Các điểm cần sửa trong mã gốc

| Vị trí | Quan sát và tác động | Cách xử lý trong bản Kaggle |
|---|---|---|
| `train.py`, cả pretrain và train | Đánh giá test mỗi epoch và chọn checkpoint bằng test Recall@20; đúng với mô tả mục 4.5 nhưng làm đánh giá lạc quan | Tách validation từ train; test chỉ dùng sau khi chọn checkpoint |
| `main.py` | Gán `'lambda1': args.lambda0`, làm đối số `--lambda1` mất tác dụng | `--reg` và `--target-reg` độc lập |
| `train.py::sample_neg_items` | Dùng scalar Tensor làm phần tử khóa DOK dạng dict; hash khác integer, có thể bỏ sót positive khi kiểm tra membership | Lấy mẫu bằng NumPy int64, tìm kiếm trên khóa đã sắp xếp; kiểm tra rejection và trường hợp hết item âm |
| `data_utils.py::load_all` | Gộp `pv_test` vào click khi huấn luyện MBA; khác nguồn dữ liệu pretrain | Giao thức mới không dùng pv_test, giữ một click-fit nhất quán giữa teacher và target |
| `main.py`, `create_adj_mat` | Cache adjacency không có fingerprint của split/nguồn cạnh | Tạo COO từ split hiện tại; không nạp adjacency NPZ cũ |
| `LightGCN.py`, `train.py` | Nhiều lần gọi forward và reg có thể lặp lan truyền toàn đồ thị trong cùng step; evaluation cũng lặp | Một encode target/step; teacher đóng băng encode một lần; evaluation encode một lần/lượt |
| `loss_func.py` | Tính log(sigmoid(x)+epsilon) dễ nhạy số học ở logits lớn | `logsigmoid` và Bernoulli KL trong không gian logit |
| `evaluate.py` | Batch có thể trở thành 0 với dưới 10 users; lưu danh sách top-k của nhiều batch; một số tham số CLI khó dùng | Batch dương, top-k theo item chunk, cộng metric rõ ràng và kiểm tra catalog |
| `requirements.txt` | Khóa PyTorch 1.10.1 / SciPy 1.7.1 không phù hợp để cài thẳng vào Kaggle mới | Dùng torch/numpy/pandas có sẵn; không cần SciPy hoặc hạ phiên bản CUDA |

Kiểm tra riêng cache Taobao hiện có: đồ thị click chứa 1,341,117 cặp tương tác, đúng union `pv_train + buy_train`. Trong khi `load_all` tạo 1,623,932 cặp sau khi gộp thêm `pv_test`. Vì vậy cache không biểu diễn tập cạnh đang được loader MBA sử dụng. Việc giữ đồ thị teacher cố định có thể là một lựa chọn thiết kế, nhưng cần được chỉ rõ và không phụ thuộc file cache có sẵn hay chưa.

Có một khác biệt ký hiệu giữa PDF và code: Eq.17/18 in dấu `+C2*t` trong kỳ vọng, còn `denoise_negative_loss(..., DP)` của code dùng `-C2*t`. Bản Kaggle theo **code thực thi** và cách hiểu negative log likelihood, tạo penalty `+C2*t` trong loss cần tối thiểu hóa. Cần ghi khác biệt này khi nói về tái lập phương trình.

## 5. Giải thuật Adaptive MBA đã hiện thực

Ký hiệu: teacher mua hàng B, teacher click V, target T. Hai teacher được tiền huấn luyện bằng BPR rồi đóng băng. Giữ bốn bộ ước lượng nhiễu MF, cập nhật luân phiên DP/DN như mã gốc.

### 5.1. Căn chỉnh click theo độ tin cậy

Từ **training-fit**, tính số cặp mua b_u, số cặp click v_u và tỷ lệ toàn cục rho. Làm trơn:

```text
conversion_u = (b_u + 20*rho) / (v_u + 20)
reliability_u = conversion_u / (conversion_u + rho)
agreement_ui = exp(-abs(sigmoid(V_ui) - sigmoid(B_ui)) / tau)
w_ui = floor + (1-floor) * reliability_u * agreement_ui
```

Mặc định tau=0.25, floor=0.2. Trọng số có cận dưới để không tắt hẳn tín hiệu phụ. Đây là **heuristic độ tin cậy**, không phải xác suất đúng được hiệu chuẩn. Teacher học BPR chỉ được đảm bảo về xếp hạng; sigmoid của teacher không tự nhiên là xác suất chuyển đổi. Vì thế cần ablation `without_adaptive_kl`, thay tau/floor theo validation và cân nhắc calibration nếu muốn diễn giải xác suất.

Loss căn chỉnh:

```text
L_align = alpha(epoch) * [KL(B || T) + w_ui * KL(V || T)]
alpha(epoch) = alpha_max * min(1, epoch / warmup_epochs)
```

Hai phần KL đều là KL giữa Bernoulli, không phải softmax toàn catalog. Hệ số temperature có T^2; mặc định temperature=1. Gating phụ thuộc user **và** item, warmup mặc định 5 epoch. Mua hàng vẫn là nguồn neo chính.

### 5.2. Thêm tín hiệu xếp hạng mua hàng

Chỉ trên **purchase-fit**, thêm:

```text
confidence_uij = floor + (1-floor) * sigmoid(B_ui - B_uj)
L_rank = confidence_uij * softplus(T_uj - T_ui)
L_total = L_denoise_DP_DN + L_align + lambda_rank*L_rank + lambda_reg*L2
```

Mặc định lambda_rank=1. Target của adaptive khởi tạo từ teacher mua hàng. Cách này giữ thông tin xếp hạng đã học và làm loss phù hợp hơn với Recall/NDCG. `without_bpr`, `without_confidence_bpr` và `without_warm_start` kiểm tra riêng ba quyết định này.

Bản adaptive dùng L2 trên embedding được lấy mẫu của target và hai denoiser đang hoạt động. Bản đối chứng `mba` giữ L2 toàn target theo thang đo nguồn. Do đó, chênh lệch giữa hai bản là **toàn bộ gói thay đổi**; không thể quy toàn bộ cải thiện cho gating KL.

### 5.3. Lấy mẫu âm và cân bằng click

Mọi negative đều được kiểm tra với **positive training**, không đối chiếu nhãn test. Với target MBA, loại cả click-fit và purchase-fit. Điều này giảm âm giả nhưng không thể đảm bảo mọi item chưa quan sát là người dùng không thích.

Adaptive dùng hỗn hợp 80% uniform + 20% sqrt-popularity, lấy bốn candidate cho mỗi purchase và chọn candidate có score target cao thứ hai. Đây là cách lấy mẫu tương đối khó trong pool nhỏ, không bảo đảm định nghĩa semi-hard theo một margin cố định. Popularity sampling có thể làm tăng thiên lệch; kiểm tra `uniform_single_negative`, coverage và mức tập trung vào item phổ biến trước khi giữ lại.

Mỗi epoch giữ toàn bộ purchase-fit; lấy ngẫu nhiên tối đa 1 click-only/purchase trên Beibei và 2 trên Taobao. Mỗi epoch lấy lại subset, không xóa vĩnh viễn click khỏi dữ liệu/teacher. So sánh `all_clicks` để xem tốc độ giảm có làm mất tín hiệu không. Loss vẫn tính theo batch và hai pha DP/DN; cap này không bảo đảm hai nhiệm vụ có gradient cân bằng hoàn toàn.

### 5.4. Trộn dự đoán theo user

```text
beta_u = clip(beta + fusion_strength*(reliability_u - 0.5), 0.5, 1.0)
score_ui = beta_u*sigmoid(T_ui) + (1-beta_u)*sigmoid(B_ui)
```

Mặc định fusion_strength=0.1, và beta xuất phát từ README tác giả. Đây là gate dựa trên thống kê, không phải attention được học. So sánh `fixed_fusion` với beta cố định. Khi nghiên cứu tác dụng riêng của training, nên dùng fusion_strength=0 cho cả hai bản.

### 5.5. Cấu hình bắt đầu và ưu tiên theo dataset

| Cấu hình MF | Beibei | Taobao |
|---|---:|---:|
| Embedding / batch / LR | 32 / 2048 / 0.001 | 32 / 2048 / 0.001 |
| alpha / C1 / C2 | 10 / 100 / 100 | 1 / 1 / 1 |
| Teacher L2 / target L2 | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| beta | 0.7 | 0.8 |
| Adaptive click-only cap / purchase-fit | 1 | 2 |
| Teacher epoch cap / target epoch cap | 100 / 100 | 100 / 100 |

Epoch cap=100 nhằm tạo một cấu hình chạy hữu hạn; nguồn mặc định 400. Nếu validation vẫn tăng ở epoch cuối, tăng giới hạn thay vì coi mô hình đã hội tụ. Early stopping sau 4 lần kiểm tra không cải thiện; kiểm tra epoch 1 và mỗi 5 epoch.

Với **Beibei**, ưu tiên giảm click lấn át purchase và kiểm tra alpha thấp: {1, 3, 10, 30}; không mặc định alpha>=100 vì Figure 3 cho thấy suy giảm lớn. Với **Taobao**, ưu tiên warm-start/teacher tốt, kiểm tra nhóm <=3 purchases, thử alpha {1, 10, 100} và BPR weight {0.1, 0.5, 1}. Đây là vùng tìm kiếm đề xuất cho loss mới, không phải thông số đã tối ưu. Tất cả quyết định dựa trên validation.

LightGCN hỗ trợ sẵn để đối chiếu. Nếu muốn phát triển tiếp, nên thử đồ thị click có trọng số theo agreement hoặc graph dropout có kiểm soát; chưa đưa vào mặc định vì chi phí và nhiễu lan truyền. Dữ liệu hiện tại không có cart/timestamp nên chưa hiện thực mô hình chuỗi, decay theo thời gian hoặc alignment ba hành vi.

## 6. Giao thức thực nghiệm và tiêu chí chấp nhận

1. Giữ nguyên `buy_test` hiện có. Tách validation theo user từ `buy_train` với split_seed=2026; mỗi user có >=2 cặp giữ tối thiểu một cặp validation và một cặp fit. `val_fraction=0.1` là mục tiêu trước làm tròn, **không bảo đảm 10% toàn tập**. Với dữ liệu hiện tại: Beibei buy-fit=216,611, buy-val=27,050; Taobao buy-fit=160,572, buy-val=48,333.
2. Click cũng tách validation từ `pv_train`; click-fit union với purchase-fit. Nếu union làm một click-validation xuất hiện lại trong graph, loại cặp đó khỏi click-validation. Không thêm purchase-val hoặc purchase-test vào graph. Click gốc có thể chứa cặp mua được giữ lại, theo giao thức auxiliary nêu trên.
3. Mọi mô hình/cấu hình/seed dùng cùng split. Từ vựng ID được xác định theo catalog transductive, bao gồm ID test; label test không tham gia loss. Lưu mapping và splits để tái kiểm tra.
4. Pretrain hai teacher; chọn checkpoint từng teacher bằng validation riêng. Target chọn bằng validation purchase Recall@20. Không dùng test để early stopping, alpha/beta search hoặc chọn ablation thắng cuộc. Code chạy test một lần sau khi chọn checkpoint cho mỗi run; nếu xem nhiều run, vẫn phải chốt siêu tham số từ validation.
5. Đánh giá full catalog bằng Recall/NDCG@10,20, macro theo user. Validation che purchase-fit. Test che toàn purchase-train đã biết, gồm purchase-val; không loại mọi item từng click khỏi ứng viên dự đoán mua hàng. Không retrain lại trên purchase-val sau chọn checkpoint, nên lượng dữ liệu học nhỏ hơn bài gốc.
6. Chạy `mba` và `adaptive` cho seeds 2020/2021/2022; thêm teacher mua hàng làm baseline đơn hành vi. Với nghiên cứu sâu, chạy các ablation trong notebook. Ghi thêm thời gian, bộ nhớ và phân nhóm ít lượt mua; file per-user chứa số purchase-train để làm phân nhóm.
7. `summarize_results.py` xuất mean/std giữa seed và bootstrap CI cho chênh lệch per-user. CI bootstrap điều kiện trên các model seed đã train, không đo toàn bộ bất định do quá trình training. Một seed không đủ ước lượng std. Không dùng CI đó để tự động tuyên bố ý nghĩa thống kê giữa các lần huấn luyện độc lập.

Chấp nhận cải tiến khi mean Recall@20 và NDCG@20 trên test tốt hơn đối chứng được chọn bằng validation, kết quả ổn định qua seed, nhóm ít lượt mua không suy giảm đáng kể và chi phí phù hợp. Không quy đổi chỉ tiêu bài gốc thành cam kết kết quả của giao thức mới. Chỉ đặt mục tiêu tăng tương đối sau khi có baseline chạy lại thực tế.

## 7. Chạy trên Kaggle và đầu ra

Import notebook, Add Input dataset chứa CSV, bật GPU, Run All. Notebook đã nhúng toàn bộ mã bằng `%%writefile`, dùng gói có sẵn trong Kaggle, không cần git clone/pip/network. Nếu tìm thấy nhiều bản của một dataset, chương trình dừng và yêu cầu đặt `DATA_ROOT` cụ thể để tránh dùng nhầm.

Mặc định notebook chạy cả hai dataset, hai phương pháp, một seed; có thể đặt ba seed và bật ablation. Teacher cache kiểm tra hash dữ liệu, split, tham số và môi trường trước khi dùng lại. Cấu hình/split/teacher trùng nhau có thể dùng chung teacher để tiết kiệm thời gian. Không có huấn luyện phân tán nhiều GPU; dùng một GPU mặc định. LightGCN còn phải lan truyền toàn graph mỗi optimizer step nên tốn nhiều hơn MF.

Mỗi run lưu `config.json`, `splits.npz`, `history.jsonl`, `teacher_buy.pt`, `teacher_pv.pt`, `target_best.pt`, `target_last.pt`, `test_per_user.npz`, `result.json`, `metrics.csv`; adaptive có thêm `beta_by_user.npy`. Kết quả tổng hợp và ZIP được tạo ở cuối notebook. Các checkpoint chứa trọng số phục vụ đánh giá/suy luận, **chưa lưu đủ optimizer/RNG để resume chính xác một run bị ngắt**. Run đã hoàn thành chỉ được dùng lại nếu cấu hình và hash data không thay đổi.

Code có thể chạy trực tiếp ngoài notebook:

```bash
python kaggle/train_mba.py --data-root /kaggle/input --dataset taobao --variant adaptive --output /kaggle/working/mba_runs
python kaggle/train_mba.py --data-root /kaggle/input --dataset beibei --variant adaptive --output /kaggle/working/mba_runs
python kaggle/summarize_results.py --root /kaggle/working/mba_runs
```

## 8. Mức kiểm chứng đã thực hiện

Đã kiểm tra phương trình KL và DP/DN, gradient qua pha hoạt động, adjacency chuẩn hóa, split không trùng, mẫu âm không thuộc training-positive, metrics theo ví dụ tính tay và top-k theo chunk khớp cách tính dense. Kiểm thử tích hợp đi qua pretrain click, pretrain purchase, target, checkpoint và evaluation cho đối chứng MF, adaptive MF và adaptive LightGCN trên dữ liệu giả lập.

Đã chạy smoke với CSV thật của **cả Beibei và Taobao**: mỗi giai đoạn một epoch, tối đa hai optimizer step, đánh giá 64 users. Luồng hoàn tất và sinh checkpoint/metadata; kết quả mang `debug=true` và bị loại khỏi bảng tổng hợp. Những số metric gần 0 trong smoke phản ánh mô hình gần như chưa học, không phải đánh giá thuật toán.

Môi trường kiểm thử cục bộ: Python 3.9.13, PyTorch 1.10.1+cpu, NumPy 1.22.4, pandas 2.3.3. Chưa chạy huấn luyện đầy đủ hoặc kiểm chứng trực tiếp GPU Kaggle trong phiên làm việc này. Mã dùng API torch phổ biến và ghi môi trường thực tế khi chạy; seed không bảo đảm bitwise deterministic cho mọi kernel sparse GPU.

## Tài liệu tham khảo

- [Xin et al., SIGIR 2023, arXiv:2305.05585v1](https://arxiv.org/abs/2305.05585v1), đặc biệt mục 3.3, 4.3, 4.5, Table 1-4 và Figure 3; PDF cục bộ: `docs/2305.05585v1.pdf`.
- [Repository tác giả MBA](https://github.com/LiuXiangYuan/MBA), README và các tệp tương ứng trong workspace. Những nhận xét lỗi code ở đây dựa trên bản cục bộ đã đọc.
- [Kaggle Docker Python](https://github.com/Kaggle/docker-python), môi trường notebook CPU/GPU; [hướng dẫn GPU của Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage).
- [PyTorch sparse matrix multiplication](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html), [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html). Code dùng state_dict chỉ có Tensor và hỗ trợ tham số weights_only khi khả dụng.
