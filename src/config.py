"""项目级常量与路径配置。

所有脚本从这里取路径与随机种子，避免各自拼接字符串。
"""

from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DOCS = ROOT / "docs"

# 数据集标识
CATEGORY = "Video_Games"
REVIEW_FILE = DATA_RAW / "Video_Games.json.gz"
META_FILE = DATA_RAW / "meta_Video_Games.json.gz"

# 官方下载地址
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
REVIEW_URL = f"{BASE_URL}/categoryFiles/{CATEGORY}.json.gz"
META_URL = f"{BASE_URL}/metaFiles2/meta_{CATEGORY}.json.gz"

# 文件校验值，用于确认本地数据与官方一致
SHA256 = {
    "Video_Games.json.gz": "E5E79BD9A666891F83CAD29BDEE3D2A5506F63C20D3B84EA35488C49D578761B",
    "meta_Video_Games.json.gz": "4069E17C7EFA33E7574C2E77F1FE0B0C6F2A42A7AC29C4422696D788D8A8E0A6",
}

# 复现约定
SEED = 42
# 分类任务的分层分组折数
N_SPLITS = 5

# 中间产物
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
FOLDS_PARQUET = DATA_PROCESSED / "folds.parquet"

# 业务常量
# 评分预测的正类划分：4 星及以上视为正向评价
POSITIVE_RATING_THRESHOLD = 4.0
# 关联规则默认阈值，最终取值需由敏感性实验确定
DEFAULT_MIN_SUPPORT = 0.005
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_MIN_LIFT = 1.0
# 共购篮为推荐位列表而非收银小票，篮长需截断，默认上限待实验确定
DEFAULT_MAX_BASKET_LEN = 50

# 图表的中文字体候选，按顺序取第一个系统已安装的
FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"]
