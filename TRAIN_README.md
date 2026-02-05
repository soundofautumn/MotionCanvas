# 训练脚本使用说明

## Windows 脚本版本

已将 Linux shell 脚本 `train_1.3b.sh` 转换为 Windows 版本：

- **train_1.3b.bat** - Windows 批处理脚本
- **train_1.3b.ps1** - PowerShell 脚本
- **train_1.3b_config.bat** - 配置文件示例

## 使用方法

### 方法 1: 使用配置文件（推荐）

所有路径现在默认配置为**项目当前目录下的相对路径**。

1. **编辑配置文件**：
   - **批处理版本**: 编辑 `train_1.3b_config.bat`
   - **PowerShell 版本**: 编辑 `train_1.3b_config.ps1`

   配置文件中的路径都是相对于项目根目录的，例如：
   ```bat
   set DATA_PATH=data\dataset
   set OUTPUT_PATH=outputs\checkpoints
   set DIT_PATH=models\wan_1.3b\dit
   set VAE_PATH=models\wan_1.3b\vae
   set TEXT_ENCODER_PATH=models\wan_1.3b\text_encoder
   set IMAGE_ENCODER_PATH=models\wan_1.3b\image_encoder
   ```

2. **运行训练脚本**（脚本会自动加载配置文件）：
   ```cmd
   train_1.3b.bat
   ```
   或
   ```powershell
   .\train_1.3b.ps1
   ```

### 方法 2: 手动设置环境变量

如果不想使用配置文件，可以手动设置环境变量：

**批处理 (CMD):**
```cmd
set DATA_PATH=data\dataset
set OUTPUT_PATH=outputs\checkpoints
set DIT_PATH=models\wan_1.3b\dit
set VAE_PATH=models\wan_1.3b\vae
set TEXT_ENCODER_PATH=models\wan_1.3b\text_encoder
set IMAGE_ENCODER_PATH=models\wan_1.3b\image_encoder
train_1.3b.bat
```

**PowerShell:**
```powershell
$env:DATA_PATH = "data\dataset"
$env:OUTPUT_PATH = "outputs\checkpoints"
$env:DIT_PATH = "models\wan_1.3b\dit"
$env:VAE_PATH = "models\wan_1.3b\vae"
$env:TEXT_ENCODER_PATH = "models\wan_1.3b\text_encoder"
$env:IMAGE_ENCODER_PATH = "models\wan_1.3b\image_encoder"
.\train_1.3b.ps1
```

### 方法 3: 使用绝对路径

如果需要使用绝对路径，直接在配置文件中设置完整路径即可：
```bat
set DATA_PATH=D:\Project\GraduationProject\MotionCanvas\data\dataset
set OUTPUT_PATH=D:\Project\GraduationProject\MotionCanvas\outputs\checkpoints
```
脚本会自动识别绝对路径，不会进行转换。

## 必需的环境变量

| 变量名 | 说明 | 必需 | 默认相对路径 |
|--------|------|------|-------------|
| `DATA_PATH` | 训练数据集路径 | ✅ | `data\dataset` |
| `OUTPUT_PATH` | 检查点保存路径 | ✅ | `outputs\checkpoints` |
| `DIT_PATH` | DiT 模型路径 | ✅ | `models\wan_1.3b\dit` |
| `VAE_PATH` | VAE 模型路径 | ✅ | `models\wan_1.3b\vae` |
| `TEXT_ENCODER_PATH` | 文本编码器路径 | ✅ | `models\wan_1.3b\text_encoder` |
| `IMAGE_ENCODER_PATH` | 图像编码器路径 | ✅ | `models\wan_1.3b\image_encoder` |
| `TXT_PATH` | 无效数据列表路径 | ❌ | `data\invalid_data.txt` |
| `RESUME_CKPT_PATH` | 恢复训练的检查点路径 | ❌ | `outputs\checkpoints\last.ckpt` |

**注意**: 
- 所有路径默认使用**相对路径**（相对于项目根目录）
- 脚本会自动将相对路径转换为绝对路径
- 如果路径已经是绝对路径（以 `\` 或 `D:` 开头），则不会转换

## 分布式训练设置

对于多节点训练，可以设置以下环境变量：

```cmd
set RANK=0                    # 当前节点排名（0, 1, 2, ...）
set MASTER_ADDR=localhost     # 主节点地址
set MASTER_PORT=29500         # 主节点端口
```

## 训练参数说明

脚本中的主要训练参数：

- `--max_epochs 10000`: 最大训练轮数
- `--learning_rate 5e-5`: 学习率
- `--batch_size 1`: 批次大小
- `--num_frames 49`: 视频帧数
- `--target_fps 15`: 目标帧率
- `--training_strategy deepspeed_stage_1`: 训练策略（DeepSpeed Stage 1）

## 注意事项

1. **虚拟环境**: 脚本会自动激活 `.venv` 虚拟环境（如果存在）
2. **路径配置**: 
   - 所有路径默认配置为项目当前目录下的相对路径
   - 脚本会自动将相对路径转换为绝对路径
   - 支持使用绝对路径（脚本会自动识别）
3. **配置文件**: 
   - 批处理脚本会自动加载 `train_1.3b_config.bat`（如果存在）
   - PowerShell 脚本会自动加载 `train_1.3b_config.ps1`（如果存在）
4. **路径格式**: Windows 路径可以使用反斜杠 `\` 或正斜杠 `/`
5. **GPU 要求**: 确保已安装 GPU 版本的 PyTorch
6. **内存要求**: 训练需要大量 GPU 内存，请根据实际情况调整 `--batch_size` 和 `--nproc_per_node`
7. **目录结构**: 建议按照以下结构组织项目：
   ```
   MotionCanvas/
   ├── data/
   │   ├── dataset/          # 训练数据集
   │   └── invalid_data.txt   # 无效数据列表
   ├── models/
   │   └── wan_1.3b/
   │       ├── dit/           # DiT 模型
   │       ├── vae/            # VAE 模型
   │       ├── text_encoder/   # 文本编码器
   │       └── image_encoder/  # 图像编码器
   └── outputs/
       └── checkpoints/       # 训练检查点
   ```

## 故障排除

### 如果遇到 "torchrun not found" 错误：

确保已安装 PyTorch 并激活虚拟环境：
```cmd
.venv\Scripts\activate
python -c "import torch; print(torch.__version__)"
```

### 如果遇到路径错误：

- 检查所有路径是否正确设置
- 确保路径中的文件/目录存在
- 使用绝对路径而不是相对路径

### 如果遇到 CUDA 错误：

- 检查 GPU 驱动是否正确安装
- 验证 PyTorch 是否支持 CUDA：`python -c "import torch; print(torch.cuda.is_available())"`
