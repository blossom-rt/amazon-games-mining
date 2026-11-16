# Streamlit Community Cloud 部署

## 目标

把 `src/dashboard_app.py` 部署到 Streamlit Community Cloud，得到一个固定域名：

```text
https://<app-name>.streamlit.app
```

GitHub 只负责存代码；Cloud 负责运行 Python 服务。GitHub Pages 只能托管静态 HTML，不能运行 Streamlit。

## 部署前必须存在的文件

| 文件 | 作用 |
| --- | --- |
| `src/dashboard_app.py` | 应用入口 |
| `requirements.txt` | Cloud 默认安装的依赖；已包含 `streamlit`、`plotly`、`pandas` |
| `results/tables/cluster_outcomes.json` | 分群视图 |
| `results/tables/classify_time.json` | 时间视图 |
| `results/tables/network_compare.json` | 网络视图 |
| `results/tables/classify_summary.json` | 总览与模型指标 |

`results/tables` 下的这些 JSON 必须提交到 GitHub。原始 gzip 与 parquet 不需要提交。

本地看板依赖也可以单独安装：

```powershell
pip install -r requirements-dashboard.txt
```

但 Community Cloud 默认读取仓库根目录的 `requirements.txt`，所以根文件里也保留了 `streamlit` 与 `plotly`。

## 推送代码

在仓库根目录确认改动并推送：

```powershell
git status --short
git add requirements.txt src/dashboard_app.py results/tables docs/streamlit_deploy.md
git commit -m "docs: add Streamlit Cloud deployment guide"
# main 受保护时，先合入 PR 再触发部署
git push origin main
```

提交前确认暂存区不含 `data/raw`、`data/processed` 或 `.parquet`。

## 创建应用

1. 打开 <https://share.streamlit.io>，用 GitHub 登录。
2. 选择 **Create app**。
3. 填写：

| 字段 | 值 |
| --- | --- |
| Repository | `blossom-rt/amazon-games-mining` |
| Branch | `main` |
| Main file path | `src/dashboard_app.py` |

4. 打开 **Advanced settings**，Python 选择 **3.11**。
5. 点击 **Deploy**，等待构建日志结束。

构建成功后地址形如：

```text
https://amazon-games-mining-dashboard.streamlit.app/
```

当前实际地址以 Cloud 页面显示的 App URL 为准。

## 更新数据后的重新部署

看板读取的是仓库里的 JSON。更新结果后：

```powershell
python src/run_all.py --extras
git add results/tables/*.json
git commit -m "chore: refresh dashboard data"
git push origin main
```

Cloud 检测到 `main` 分支更新后会自动重新部署。若页面仍是旧数据，在应用右上角菜单选择 **Reboot app**。

## 运行限制

- 免费应用空闲一段时间后会休眠，再次打开需要等待唤醒。
- 免费额度有内存和运行时长限制，不适合长期高并发。
- 应用默认公开；页面只有聚合结果，没有评论人姓名、reviewerID 等个人信息。
- 当前 `dashboard_app.py` 不读取 parquet，只读四个 JSON，启动快、内存小。

## 常见问题

**构建时报 `ModuleNotFoundError: streamlit`**

根目录 `requirements.txt` 没有 `streamlit`，或应用连到了错误分支。确认文件中有：

```text
streamlit==1.67.0
plotly==7.1.0
```

**构建时报 `ModuleNotFoundError: pandas`**

根目录 `requirements.txt` 缺少 `pandas`。本项目固定 `pandas==3.1.1`。

**页面报 `FileNotFoundError`**

四个 JSON 没有提交到 GitHub，或分支不是预期的 `main`。检查：

```powershell
git ls-files results/tables/cluster_outcomes.json results/tables/classify_time.json results/tables/network_compare.json results/tables/classify_summary.json
```

**构建超时**

根 `requirements.txt` 包含研究用的 shap、lime、jupyterlab 等重依赖。若 Cloud 构建超时，可建一个仅用于部署的分支，根 `requirements.txt` 只保留：

```text
pandas==3.1.1
numpy==2.4.6
streamlit==1.67.0
plotly==7.1.0
```

把应用连接到该分支，数据 JSON 一并带过去。

**应用一直转圈**

先访问 `https://<app-name>.streamlit.app/_stcore/health`。返回 `ok` 说明服务已启动；刷新浏览器或 Reboot。企业代理可能拦截 WebSocket。

## 备选方案

- **静态版**：`results/dashboard.html` 由 `src/build_static_dashboard.py` 生成，四个视图与两个联动筛选都在前端完成。合入 GitHub 后可用 raw.githack 或 GitHub Pages 访问。
