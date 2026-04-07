

import os
import cv2
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import BeitFeatureExtractor, BeitForMaskedImageModeling

# =======================
# 1) Dataset: 拆解影片幀
# =======================
class VideoFrameDataset(Dataset):
    """
    讀取單一 mp4 檔，將影片逐幀讀入後，每次 __getitem__ 回傳一張影像。
    實務中若影片太長，建議做 streaming / 多影片 dataset / 多處理等。
    """
    def __init__(self, video_path, transform=None):
        super().__init__()
        self.video_path = video_path
        self.transform = transform

        # 讀取影片所有幀
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        while success:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)
            success, frame = cap.read()
        cap.release()

        if len(self.frames) == 0:
            raise ValueError(f"video {video_path} can't read or frame is zero!")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        if self.transform:
            img = self.transform(img)  # transform 必須回傳 [C, H, W] 的 Tensor
        return img


# ==============================
# 2) Random Mask 的輔助函式
# ==============================
def generate_bool_masked_pos(size, mask_ratio=0.5):
    """
    產生隨機 mask, BEiT / DiT MIM 會針對 'patch' 做遮蔽，此處示範簡單 random。
    注意：實務中最好配合模型之 patch granularity 做更精細對齊。
    Args:
        size (int): token / patch 的總數 (視 feature_extractor 設定而定)
        mask_ratio (float): 有多少比例要被 mask
    Return:
        torch.BoolTensor, shape=[size], True 代表該 token/patch 被 mask
    """
    num_mask = int(mask_ratio * size)
    mask = [False] * size
    # 隨機挑選 num_mask 個位置標為 True
    mask_idx = random.sample(range(size), num_mask)
    for m in mask_idx:
        mask[m] = True
    mask = torch.BoolTensor(mask)
    return mask


# ============================
# 3) 訓練函式
# ============================
def train_one_epoch(model, dataloader, feature_extractor, optimizer, device, mask_ratio=0.5):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # batch shape: [B, C, H, W], 來自 transforms.ToTensor()
        images = batch.to(device)

        # 將圖像轉成模型可用的 pixel_values
        # feature_extractor 通常會回傳: {'pixel_values': (B, 3, H', W')}
        # 同時我們需要 'bool_masked_pos' => shape為 (B, num_patches)
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]  # (B, 3, H', W')

        # 取得該模型對應的patch數量
        # BEiT base, patch_size=16 => 每張224x224圖像約14x14=196個patch
        # 這裡我們用 pixel_values.shape 可能無法直接得知 patch 數，故查看官方: num_patches = (image_size // patch_size)^2
        # 也可先把 image_size = feature_extractor.size, patch_size = 16 => num_patches=(224//16)^2=196
        # 或者在 forward 時會自動計算 patch embedding
        batch_size = pixel_values.shape[0]

        # 這裡簡單 assume 預設模型 config 中 image_size=224, patch_size=16 => 196個patch
        # 實際可從 model.config.num_patches 取得(beit沒定義num_patches, 要手動推)
        num_patches = 196  # 預設 mictosoft/dit-base config
        bool_masked_pos = []
        for _ in range(batch_size):
            mask = generate_bool_masked_pos(num_patches, mask_ratio=mask_ratio)
            bool_masked_pos.append(mask)
        bool_masked_pos = torch.stack(bool_masked_pos, dim=0).to(device)

        # forward
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================
# 4) 主程式：finetune.py
# ============================
def main():
    # ---------------------------
    # 基本參數設定
    # ---------------------------
    video_path = "output/result_9.mp4"   # 影片路徑
    batch_size = 4
    num_workers = 2
    epochs = 2
    lr = 1e-4
    mask_ratio = 0.5        # 有 50% patch 被 mask
    img_size = 224          # 與 microsoft/dit-base 相容 (預設224)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # 1. 建立 Dataset / DataLoader
    # ---------------------------
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),  # 跟 dit-base 預設 224 相符
        transforms.ToTensor(),
    ])
    dataset = VideoFrameDataset(video_path=video_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # ---------------------------
    # 2. 載入預訓練模型 & feature_extractor
    # ---------------------------
    print("Loading pretrained DiT (BeitForMaskedImageModeling) from microsoft/dit-base ...")
    feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/dit-base")
    model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
    model.to(device)

    # ---------------------------
    # 3. 準備 Optimizer
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---------------------------
    # 4. 訓練迴圈
    # ---------------------------
    for epoch in range(1, epochs + 1):
        print(f"=== Epoch [{epoch}/{epochs}] ===")
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            device=device,
            mask_ratio=mask_ratio
        )
        print(f"  [Train] loss: {avg_loss:.4f}")

    # ---------------------------
    # 5. 儲存微調後的模型
    # ---------------------------
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/finetuned_dit_base_mim.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Finetuning completed! Model weights saved => {save_path}")


if __name__ == "__main__":
    main()