"""Build a self-contained Kaggle notebook from the reviewed Python sources."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    result = dict(cell_type=kind, id=hashlib.sha256(text.encode()).hexdigest()[:12], metadata={},
                  source=text.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    cells = [cell("markdown", """# MBA và cải tiến thích nghi trên Taobao / Beibei

Notebook độc lập, chứa toàn bộ mã nguồn. Không cần clone repo hoặc bật Internet.

1. Upload `mba_taobao_beibei_data.zip` thành một Kaggle Dataset, rồi **Add Input** vào notebook.
   Dataset cần các CSV gốc đã giải nén: `{taobao,beibei}_{buy_train,buy_test,pv_train}.csv`.
   Các file mang đuôi CSV trong repo thực tế phân cách bằng tab và có header; mã tự nhận diện.
2. Import notebook này, bật **Settings > Accelerator > GPU** rồi **Run All**.
3. Cấu hình bên dưới mặc định chạy đối chứng MBA và bản adaptive cho cả hai dataset, một seed.
   Để báo cáo nghiên cứu, đặt `SEEDS = [2020, 2021, 2022]` và dành đủ thời gian GPU.
4. Kết quả, checkpoint, ánh xạ ID và splits nằm ở `/kaggle/working/mba_runs`.

Bản `mba` giữ mục tiêu và hai pha DP/DN của mã gốc, nhưng dùng validation an toàn và sampler đã sửa.
Đây không phải bản tái lập nguyên xi Table 2: bài gốc dùng test để early stopping.
Bản `adaptive` là đề xuất cần kiểm chứng. Notebook không chứa số liệu cải thiện được giả định sẵn.
Click-test không được đưa vào huấn luyện. Các split không có timestamp nên không chứng minh đánh giá theo thời gian.

Tài liệu phân tích đi kèm: `docs/phan_tich_cai_tien_MBA.md`.
Nguồn: [bài báo](https://arxiv.org/abs/2305.05585), [mã tác giả](https://github.com/LiuXiangYuan/MBA).
"""), cell("code", """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'NumPy:', np.__version__, 'pandas:', pd.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
REQUIRE_GPU = True
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError('Bật GPU trong Settings > Accelerator rồi chạy lại. Có thể đặt REQUIRE_GPU=False để thử CPU.')
""")]
    for name in ("train_mba.py", "summarize_results.py"):
        source = (ROOT / "kaggle" / name).read_text(encoding="utf8")
        cells.append(cell("code", f"%%writefile /kaggle/working/{name}\n" + source))
    cells.append(cell("code", "%%writefile /kaggle/working/LICENSE\n" + (ROOT / "LICENSE").read_text(encoding="utf8")))
    cells.append(cell("code", """# Cấu hình thí nghiệm
DATA_ROOT = '/kaggle/input'
RUN_ROOT = '/kaggle/working/mba_runs'
TEACHER_CACHE = '/kaggle/working/mba_teacher_cache'
DATASETS = ['taobao', 'beibei']
VARIANTS = ['mba', 'adaptive']
SEEDS = [2020]  # Báo cáo nghiên cứu: [2020, 2021, 2022]
PRETRAIN_EPOCHS = 100
TRAIN_EPOCHS = 100
BACKBONE = 'mf'  # 'lightgcn' tốn nhiều thời gian hơn; MF tốt hơn trong kết quả MBA gốc.
SMOKE_ONLY = False  # True: chỉ kiểm tra luồng chạy, KHÔNG đo chất lượng.
RUN_ABLATIONS = False

sys.path.insert(0, '/kaggle/working')
import importlib
import train_mba
importlib.reload(train_mba)
for dataset in DATASETS:
    print(dataset, train_mba.discover_data(DATA_ROOT, dataset))
"""))
    cells.append(cell("code", """import json

def execute(dataset, variant, seed, label=None, extra=()):
    argv = ['--data-root', DATA_ROOT, '--output', RUN_ROOT, '--teacher-cache', TEACHER_CACHE,
            '--dataset', dataset, '--variant', variant, '--backbone', BACKBONE,
            '--seed', str(seed), '--pretrain-epochs', str(PRETRAIN_EPOCHS),
            '--epochs', str(TRAIN_EPOCHS), '--device', 'auto']
    if label:
        argv += ['--run-label', label]
    if SMOKE_ONLY:
        argv += ['--pretrain-epochs', '1', '--epochs', '1', '--max-steps', '2', '--eval-user-limit', '64']
    argv += list(extra)
    config = train_mba.resolve_config(train_mba.parser().parse_args(argv))
    method = config.run_label or config.variant
    folder = Path(RUN_ROOT) / dataset / f'{method}_{BACKBONE}_seed{seed}'
    if (folder / 'result.json').is_file():
        saved = json.loads((folder / 'config.json').read_text())['config']
        if saved != vars(config):
            raise RuntimeError(f'Cấu hình đã đổi: hãy đặt RUN_ROOT mới để giữ kết quả cũ: {folder}')
        # Verify data too before reusing a completed run.
        old_sources = json.loads((folder / 'config.json').read_text())['sources']
        current_paths = train_mba.discover_data(DATA_ROOT, dataset)
        if any(train_mba.file_hash(current_paths[k]) != v['sha256'] for k, v in old_sources.items()):
            raise RuntimeError('Dữ liệu đã đổi; hãy dùng RUN_ROOT mới.')
        print('Đã hoàn thành, dùng lại:', folder)
        return json.loads((folder / 'result.json').read_text())
    return train_mba.run(config)

results = []
for dataset in DATASETS:
    for seed in SEEDS:
        for variant in VARIANTS:
            results.append(execute(dataset, variant, seed))
pd.DataFrame([dict(dataset=r['dataset'], method=r['method'], seed=r['seed'], debug=r['debug'], **r['test'])
              for r in results])
"""))
    cells.append(cell("markdown", """## Ablation tùy chọn

Các cấu hình dưới đây bỏ từng thành phần của bản adaptive. Mọi mô hình dùng chung split và seed.
Teacher được dùng lại chỉ khi checksum dữ liệu, cấu hình tiền huấn luyện và môi trường đều trùng khớp.
Không chọn cấu hình theo test; dùng `best_validation` trong `result.json` để quyết định cấu hình.
Chạy lại notebook với cấu hình mới cần một `RUN_ROOT` mới. Checkpoint hiện tại phục vụ đánh giá/suy luận;
chưa có cơ chế tiếp tục chính xác optimizer của một epoch bị ngắt.
"""))
    cells.append(cell("code", """ABLATIONS = {
    'without_adaptive_kl': ['--no-adaptive-kl'],
    'without_bpr': ['--bpr-weight', '0'],
    'without_confidence_bpr': ['--no-confidence-bpr'],
    'uniform_single_negative': ['--negative-candidates', '1', '--popularity-mix', '0'],
    'all_clicks': ['--click-ratio', '-1'],
    'fixed_fusion': ['--fusion-strength', '0'],
    'without_warm_start': ['--no-warm-start'],
}
if RUN_ABLATIONS:
    for dataset in DATASETS:
        for seed in SEEDS:
            for label, extra in ABLATIONS.items():
                execute(dataset, 'adaptive', seed, label, extra)
"""))
    cells.append(cell("code", """from summarize_results import summarize
summarize(RUN_ROOT)
# all_runs.csv: kết quả từng seed
# mean_std.csv: mean/std giữa các seed (một seed không đủ ước lượng std)
# paired_differences.csv: chênh lệch và bootstrap CI theo user, điều kiện trên các seed đã train
# Các lượt SMOKE_ONLY/debug tự động bị loại khỏi bảng nghiên cứu.
"""))
    cells.append(cell("code", """# Gói kết quả để tải về. Dùng Save Version để Kaggle giữ notebook output.
import shutil
archive = shutil.make_archive('/kaggle/working/mba_results', 'zip', RUN_ROOT)
print(archive)
from IPython.display import FileLink, display
display(FileLink(Path(archive).name))
"""))
    notebook = dict(cells=cells, metadata={
        "kernelspec": dict(display_name="Python 3", language="python", name="python3"),
        "language_info": dict(name="python", version="3.10.0"),
        "accelerator": "GPU",
    }, nbformat=4, nbformat_minor=5)
    destination = ROOT / "kaggle" / "MBA_Taobao_Beibei_Kaggle.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(destination)


def package(include_data):
    output = ROOT / "output" / "kaggle"
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output / "mba_kaggle_code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "kaggle").glob("*")):
            if path.suffix in (".py", ".ipynb", ".md"):
                archive.write(path, path.relative_to(ROOT))
        for path in [ROOT / "LICENSE", ROOT / "docs" / "phan_tich_cai_tien_MBA.md", ROOT / "docs" / "data_audit.json"]:
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
    if include_data:
        with zipfile.ZipFile(output / "mba_taobao_beibei_data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for dataset in ["taobao", "beibei"]:
                for path in sorted((ROOT / "data" / dataset).glob("*.csv")):
                    archive.write(path, path.relative_to(ROOT / "data"))
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-data", action="store_true", help="Also package the local CSV datasets for upload.")
    args = p.parse_args()
    build()
    package(args.include_data)
