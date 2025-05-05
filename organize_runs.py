import os
import shutil
from datetime import datetime

def organize_runs(root_dir="runs"):
    """
    將 runs/ 資料夾內的 train_ / test_ / test_many_ 類別自動分類到對應資料夾，
    並將其壓縮封存至 archive/，同時紀錄操作日誌。
    """
    folders = sorted(os.listdir(root_dir))

    # 對應分類規則
    category_map = {
        "train_": "train",
        "test_": "test",
        "test_many_": "test_many"
    }

    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = os.path.join(root_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 判斷分類
        for prefix, category in category_map.items():
            if folder.startswith(prefix):
                target_dir = os.path.join(root_dir, category)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, folder)

                # 避免覆蓋
                if not os.path.exists(target_path):
                    shutil.move(folder_path, target_path)
                    log_lines.append(f"[{timestamp}] ✅ 移動 {folder} → {category}/")
                else:
                    log_lines.append(f"[{timestamp}] ⚠️ 已存在 {category}/{folder}，略過")

                # 壓縮封存
                zip_name = f"{folder}_{timestamp}.zip"
                zip_base = os.path.join(archive_dir, zip_name.replace(".zip", ""))
                shutil.make_archive(zip_base, 'zip', target_path)
                log_lines.append(f"[{timestamp}] 📦 封存 {zip_name}")
                break  # 分類完成後退出內層迴圈

    # 寫入整理日誌
    log_path = os.path.join(root_dir, "整理日誌.log")
    with open(log_path, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(f"✅ 整理完成，共記錄 {len(log_lines)} 項操作。詳見：{log_path}")

if __name__ == "__main__":
    organize_runs()

