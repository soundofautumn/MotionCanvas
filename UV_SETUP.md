# 使用 uv 配置项目环境

本项目已使用 `uv` 配置完成。

## 环境信息

- Python 版本: 3.10.6
- 虚拟环境位置: `.venv/`
- 项目已安装为可编辑模式

## 激活虚拟环境

### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### Windows (CMD)
```cmd
.venv\Scripts\activate.bat
```

### Linux/macOS
```bash
source .venv/bin/activate
```

## 常用命令

### 安装依赖
```bash
uv pip install -e .
```

### 安装新的包
```bash
uv pip install <package-name>
```

### 更新依赖
```bash
uv pip install --upgrade -e .
```

### 查看已安装的包
```bash
uv pip list
```

### 运行项目
激活环境后，可以直接运行项目脚本：
```bash
python apps/gradio/DiffSynth_Studio.py
```

## 注意事项

- 虚拟环境目录 `.venv/` 已添加到 `.gitignore`
- 项目使用 `pyproject.toml` 管理依赖
- 所有依赖已自动安装，包括：
  - PyTorch 2.9.1
  - Transformers 4.57.3
  - Diffusers 0.36.0
  - Gradio 6.2.0
  - 以及其他必要的依赖包

