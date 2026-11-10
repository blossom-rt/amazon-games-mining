# 数据来源说明

## 数据集

Amazon Review Data (2018)，Video_Games 品类。

| 项目 | 内容 |
|---|---|
| 数据集主页 | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ |
| 备用域名 | https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/ |
| 发布方 | University of California, San Diego，McAuley Lab |
| 版本 | 2018 版（上一版为 2014 版） |
| 时间跨度 | 1996 年 5 月 — 2018 年 10 月 |
| 全量规模 | 2.331 亿条评论，29 个品类 |
| 本研究所用品类 | Video_Games，评论 2,565,349 条，商品 84,893 件 |
| 获取日期 | 2026-11-09 |

数据主页在 2020 年 8 月更新过元数据，声明「大幅减少了 HTML/CSS 代码」。本研究的文件为更新后的版本，仍残留少量 HTML 片段，这一点在数据治理章节中会单独说明。

## 本地文件

放在 `data/raw/`，不纳入版本控制。

| 文件名 | 大小 | 记录数 | SHA256 |
|---|---|---|---|
| `Video_Games.json.gz` | 493.7 MB | 2,565,349 | `E5E79BD9A666891F83CAD29BDEE3D2A5506F63C20D3B84EA35488C49D578761B` |
| `meta_Video_Games.json.gz` | 50.8 MB | 84,819 | `4069E17C7EFA33E7574C2E77F1FE0B0C6F2A42A7AC29C4422696D788D8A8E0A6` |

商品数官方标称 84,893，本地实际可解析记录 84,819，差值 74 条为重复或无法解析的记录，处理时按 `asin` 去重。

## 下载方式

```
python src/download.py
```

只校验不下载：

```
python src/download.py --verify
```

## 格式

gzip 压缩的 JSON Lines，一行一个对象，UTF-8 编码。

评论文件与元数据文件通过 `asin` 字段关联，这是唯一可靠的连接键。
注意 `reviewerID` 是匿名化后的用户编号，同一自然人的多个账号无法识别。

## 引用格式

若报告或任何衍生成果使用本数据，按官方要求引用：

> Ni J., Li J., McAuley J. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of EMNLP-IJCNLP, 2019.

## 使用限制

数据仅限学术研究使用，不得用于商业用途，不得尝试还原评论者身份。
