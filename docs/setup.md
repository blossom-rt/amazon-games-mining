# 环境配置指南

本文件面向第一次使用本仓库的读者，按顺序执行即可。不含下载数据的时间，大约需要 10 到 20 分钟。

## 前置条件

- Windows 10 或 11，macOS 与 Linux 的命令基本相同
- 已安装 Git 与 Miniconda 或 Anaconda，终端里 `conda --version` 能输出版本
- 磁盘至少 5 GB 空闲，建议 10 GB。原始数据约 545 MB，后续生成的 parquet 还会占 1 到 2 GB
- 能访问 GitHub 与 PyPI；下载原始数据还需要访问 `mcauleylab.ucsd.edu`
- 不需要 GPU

## 克隆仓库

```
git clone https://github.com/blossom-rt/amazon-games-mining.git
cd amazon-games-mining
```

数据文件与虚拟环境都不在仓库里，克隆下来只有代码和文档。

## 创建并激活环境

Windows PowerShell 第一次使用 conda 时需要初始化，只需做一次：

```
conda init powershell
```

关闭终端重新打开，然后执行：

```
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
python -V
```

最后一行应输出 `Python 3.11.x`。环境建在 conda 默认的 envs 目录即可，不需要与开发环境使用相同的绝对路径。

如果用的是 Git Bash，激活命令是 `source activate amazon-games`；cmd 里仍然是 `conda activate amazon-games`。终端提示符前出现 `(amazon-games)` 才算激活成功。

## 安装依赖

```
pip install -r requirements.txt
```

`requirements.txt` 已固定为实测版本，不要手动改版本号。安装完成后核对一下：

```
pip list | findstr /R "pandas numpy pyarrow scipy scikit-learn mlxtend matplotlib seaborn shap lime tqdm jupyterlab ipykernel"
```

PowerShell 没有 `findstr` 的话，直接 `pip list` 再对照下面的表。期望版本：

| 包 | 版本 |
|---|---|
| Python | 3.11.18（实测，3.11.x 均可） |
| pandas | 3.1.1 |
| numpy | 2.4.6 |
| pyarrow | 26.0.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| mlxtend | 0.25.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| shap | 0.51.0 |
| lime | 0.2.0.1 |
| tqdm | 4.70.1 |
| jupyterlab | 4.6.6 |
| ipykernel | 7.4.0 |

如果默认的 pip 镜像返回 403 或提示找不到版本，换成官方源重试：

```
pip install -r requirements.txt -i https://pypi.org/simple
```

国内网络也可以试阿里云镜像：

```
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

## 注册 Jupyter 内核

Jupyter 不会自动使用当前 conda 环境，需要把环境注册为内核：

```
python -m ipykernel install --user --name amazon-games --display-name "Python (amazon-games)"
```

启动：

```
python -m jupyter lab
```

打开 notebook 后，在菜单里选 Kernel → Change kernel → `Python (amazon-games)`。用 VS Code 的话，把解释器切换为 `...\envs\amazon-games\python.exe`。

不注册的后果是 notebook 仍然使用 base 环境，`import pandas` 直接报 `ModuleNotFoundError`。

## 自检

先确认依赖都能导入：

```
python -c "import pandas, numpy, pyarrow, scipy, sklearn, mlxtend, matplotlib, seaborn, shap, lime; print('依赖导入正常')"
```

再核对三个关键版本：

```
python -c "import pandas, sklearn, shap; print(pandas.__version__, sklearn.__version__, shap.__version__)"
```

应输出 `3.1.1 1.9.1 0.51.0`。

## 准备数据

两个压缩文件共约 545 MB，不纳入版本控制，需要自己获取。两种方式选一种。

方式一，用脚本下载。脚本会自动校验 SHA256，已存在且校验通过的文件不会重复下载：

```
python src/download.py
```

只想检查本地文件、不下载：

```
python src/download.py --verify
```

在项目根目录执行，看到两行 `[校验通过]` 才算成功。脚本不支持断点续传，下载中断后重新执行会从头开始。

方式二，使用已有数据副本。把两个文件放进 `data/raw/`，文件名必须完全一致：

```
data/raw/Video_Games.json.gz        493.7 MB
data/raw/meta_Video_Games.json.gz    50.8 MB
```

拷贝完仍然要跑一次 `python src/download.py --verify` 校验。

对应的 SHA256，手动核对时用得到：

```
Video_Games.json.gz       E5E79BD9A666891F83CAD29BDEE3D2A5506F63C20D3B84EA35488C49D578761B
meta_Video_Games.json.gz  4069E17C7EFA33E7574C2E77F1FE0B0C6F2A42A7AC29C4422696D788D8A8E0A6
```

## 冒烟测试

数据就位后，读一条商品元数据确认文件可解析：

```
python -c "from src.io_utils import peek_jsonl_gz; from src.config import META_FILE; r = peek_jsonl_gz(META_FILE, 1)[0]; print(r.get('asin'), '|', r.get('title'))"
```

能打印出一个 ASIN 和商品标题，说明环境与数据都已就绪。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `conda` 不是内部或外部命令 | 用开始菜单里的 Anaconda Prompt；或执行 `conda init powershell` 后重开终端 |
| `pip install` 报 403 或 `from versions: none` | 镜像不可用，加 `-i https://pypi.org/simple` |
| notebook 里 `ModuleNotFoundError: pandas` | 内核没注册或选错，回到“注册 Jupyter 内核”一节 |
| 图表中文显示成方块 | 系统缺中文字体。Windows 安装 `SimHei` 或 `Microsoft YaHei`；Linux 与 macOS 安装 `Noto Sans CJK`，必要时改 `src/config.py` 里的 `FONT_CANDIDATES` |
| 下载很慢或连接超时 | 使用已有数据副本；访问境外站点通常较慢 |
| SHA256 不一致 | 删除对应文件重新下载，不要用浏览器另存后改名的副本 |
| `pandas` 版本不是 3.1.1 | 多半在 base 环境里装了，重新激活 `amazon-games` 并重装依赖 |
| shap 导入时报 numba 或 llvmlite 错误 | 确认 numpy 是 2.4.6；必要时 `pip install --force-reinstall -r requirements.txt` |
| 解析时内存不足 | 推荐 16 GB 以上内存；解析脚本按块写入，不会一次读入整个文件 |

## 工作目录与约定

- 所有命令在项目根目录执行；脚本用 `python src/脚本名.py`，也可以用 `python -m src.脚本名`
- 随机种子统一取 `src/config.py` 里的 `SEED = 42`
- 数据、虚拟环境、缓存都不入库，不要 `git add data/` 或 `git add .conda/`
- 分支约定：`main` 保持稳定，日常开发在 `dev` 上提交，阶段完成后开 PR 合并
