# Multimodal-yolo
# **📝 自訂 Function 及 修改內容整理**
## **📌 新增 Function**
| **Function 名稱** | **功能** | **所在檔案** |
|-------------------|---------|-------------|
| `_setup_fusion_img(batch, fusion_path)` | 透過影像路徑讀取融合影像，並將 `batch["fusion_path"]` 轉換為 list | `trainer.py` |
| `_fusion_process(batch)` | 讀取 `batch["fusion_path"]` 的影像，將其轉換為 PyTorch Tensor 並存入 `batch["fusion_tensor"]` | `trainer.py` |
| `BaseModel.forward(x, x2=None, *args, **kwargs)` | 修改 `forward()` 讓 `x2` (fusion_tensor) 參與前向傳播 | `nn/tasks.py` |
| `MultiConv.forward(x, x2)` | 在 `MultiConv` 中加入 `fusion_tensor` 的 maxpool、conv 運算，並確保尺寸匹配 | `nn/modules/conv.py` |

---

## **🛠 修改項目**
| **修改項目** | **修改內容** | **所在檔案** |
|-------------|------------|-------------|
| **trainer.py** `_do_train()` | 確保 `batch["fusion_tensor"]` 被正確載入並傳遞到 `self.model()` | `trainer.py` |
| **trainer.py** `_setup_fusion_img()` | 確保 `batch["fusion_path"]` 是 list，避免 `NoneType` 錯誤 | `trainer.py` |
| **trainer.py** `_fusion_process()` | 確保 `batch["fusion_tensor"]` 形狀一致，避免 `RuntimeError` | `trainer.py` |
| **BaseModel.forward()** | 讓 `batch["fusion_tensor"]` 參與 `predict()`，並在 `x2=None` 時複製 `x` | `nn/tasks.py` |
| **MultiConv.forward()** | 確保 `x` 和 `x2` 經過 pooling、conv 後仍然匹配，避免 `torch.cat()` shape 錯誤 | `nn/modules/conv.py` |
| **nn/model.py**|於`__call__``predict()`中加入參數x2|確保融合圖像能夠正確傳輸|
|**nn/task.py**|於`predic``_predict_once`中加入x2參數|確保融合圖像能夠正常傳遞|
|**engine/predictor**|於`inference`中加入x2參數|確保融合圖像得以正常傳遞|
|**engine/predictor**|於`preprocess`中加入檢查輸入參數是否為Tensor與維度|確保所有輸入皆以Tensor傳遞|
|**nn/autobackend.py**|於`forward`中加入x2的相關處理|確保x2作為Tensor時能夠與im(原生YOLO的圖像輸入)相同|
|**engine/predictor**|於`stream_inference`中實現雙圖像的預測|透過`self.inference`預測與`postprocess`提出結果|




---

## **📌 主要修改的影響**
1. **在 `trainer.py` 確保 `batch["fusion_tensor"]` 被正確載入**  
   - `_setup_fusion_img()` 處理 `fusion_path`
   - `_fusion_process()` 讀取影像並轉為 Tensor
   - 在 `_do_train()` 內確保 `batch["fusion_tensor"]` **成功傳遞給 `self.model()`**

2. **在 `BaseModel.forward()` 讓 `fusion_tensor` 正確傳遞**
   - 訓練模式時，從 `batch` 讀取 `batch["fusion_tensor"]`
   - 預測模式時，確保 `x2` 存在，否則複製 `x`

3. **在 `MultiConv.forward()` 確保拼接時 shape 匹配**
   - 先 `maxpool(x)`, `maxpool(x2)`
   - 確保 `x.shape == x2.shape`
   - `torch.cat([x, x2], dim=1)` 之前確保尺寸一致
4. **autobackend.forward()確保 支援x2 **
   - 將融合張量的格式與目標張量一致
   - 修改self.model傳遞x2參數
5. **predictor新增x2**
   -`inference()、forward()、predict()、predict_cli()`加入x2參數
   -指定one2one傳遞於postprocess
6. **task.py 新增x2**
   -`_predict_once`
   -`predict`
   -`forward`

這些修改確保 `batch["fusion_tensor"]` **可以正常傳遞**，避免 shape mismatch 和 RuntimeError，讓 `fusion_tensor` **成功參與 YOLO 訓練和推理**。🚀
