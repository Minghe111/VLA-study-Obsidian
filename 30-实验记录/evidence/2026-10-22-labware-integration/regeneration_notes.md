# 器具资产重建说明

记录日期：2026-10-22。本文件仅说明未来如何复现；本次没有重建、重装或改写已验收的 prepared、interface、annotations。

## 固定输入与运行时

已验证使用的解释器：

`/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python`

版本：Python **3.10.21**、NumPy **1.26.4**、trimesh **4.4.3**、SciPy **1.10.1**。`prepare_assets.py` 不需要 Pillow、SAPIEN、USD Python 包或 CADQuery；烧杯和锥形瓶直接读取先前从 USD 转出的、随证据保留的 OBJ。Gmsh **4.14.1** 只用于两份现成 STEP 的表面离散化，隔离目录固定为 `/tmp/labware-cad-runtime`。

源版本：

- LabUtopia 数据 revision `e622f0ec8123c9098d913ea56a4116fd83a29662`。
- revenexx/sample-data commit `e7217fb87daf42adc2e61233a4a1bba61adef946`。
- 药瓶物理规格以本目录的 `physics_spec.json` 为输入，验收 SHA256 为 `9d5ae8d89d60dd2d4e466a966c3862d7d160cc2e5ebe4d61bc8a1eac9a2d59ef`；瓶盖质量是 0.012 kg。
- 许可、署名、转换范围参见 `LICENSES.md`。不要将生成的刻度线当作毫升计量标定。

## 临时 Gmsh 目录丢失后的恢复命令

以下只写 `/tmp`，不向系统或现有 Conda 环境安装包。需要网络的只有 `pip download`，也可将同一版本 wheel 预先放到 `/tmp/labware-gmsh-wheel/` 后略过下载。

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python -m pip download \
  --no-deps --dest /tmp/labware-gmsh-wheel gmsh==4.14.1

/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python - <<'PY'
from pathlib import Path
import hashlib
import zipfile

wheel = Path('/tmp/labware-gmsh-wheel/gmsh-4.14.1-py2.py3-none-manylinux_2_24_x86_64.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '95b3e40b38e97db6424a602ac016a9db90012b2134c336b919b4fe2e31c65e4e'
runtime = Path('/tmp/labware-cad-runtime')
runtime.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(wheel) as package:
    (runtime / 'gmsh.py').write_bytes(package.read('gmsh.py'))
    (runtime / 'libgmsh.so.4.14').write_bytes(
        package.read('gmsh-4.14.1.data/data/lib/libgmsh.so.4.14'))
assert hashlib.sha256((runtime / 'libgmsh.so.4.14').read_bytes()).hexdigest() == '9b6ba60833668f0b36ed503904a6c14032f8885916449d5fe70b43ca3cff4631'
print('Temporary Gmsh runtime ready:', runtime)
PY
```

必须将 `libgmsh.so.4.14` 放在 `gmsh.py` 同目录：本机曾遇到 `pip install --target` 仅保留 Python 包但共享库相对路径丢失的问题，所以这里显式从固定 wheel 提取两项。

系统共享库是本机现有运行时依赖，可先用 `ldd /tmp/labware-cad-runtime/libgmsh.so.4.14` 检查；若出现 `not found`，该主机尚不具备同等运行时，不能用包安装成功冒充转换可运行。本记录不包含系统软件安装动作。

## 在独立候选目录重建，保留现有验收证据

`prepare_assets.py` 以自身目录作为输出根，会重写该根下 `prepared/`、`asset_interface.json`、`source_manifest.json`。它不会生成或覆盖 `model_data0.json`，也不会自动更新已冻结的标注验收记录。因此不要直接在当前验收目录运行重建。

当前配方仍读取相邻旧证据目录中的 `converted/models.json` 和 `source/source_manifest.json`，以保留原 USD 提取的尺度、轴向和哈希链。只需下面列出的两件 OBJ 和两件 USD，不需要 USD 运行时，不需要重新下载整套 LabUtopia。现有 `sources/LabUtopia/` 中也保留了这四份原输入；下面按当前配方的目录依赖复制。

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python - <<'PY'
from pathlib import Path
import shutil

vault = Path('/home/admin123/liminghe/VLA-WAM-RL')
original = vault / '30-实验记录/evidence/2026-10-22-labware-integration'
old = original.parent / '2026-10-21-realman-lab-specific'
candidate_parent = Path('/tmp/labware-rebuild-20261022')
assert not candidate_parent.exists(), 'Use a fresh candidate directory; do not overwrite an earlier rebuild.'
candidate = candidate_parent / original.name
candidate.mkdir(parents=True)
for name in ['prepare_assets.py', 'physics_spec.json', 'LICENSES.md']:
    shutil.copy2(original / name, candidate / name)
shutil.copytree(original / 'sources', candidate / 'sources')

old_candidate = candidate_parent / old.name
for relative in [
    'converted/models.json',
    'converted/beaker_000.obj',
    'converted/erlenmeyer_000.obj',
    'source/source_manifest.json',
    'source/Instruments/Beaker_01/Beaker_01.usd',
    'source/Instruments/ErlenmeyerFlask/ErlenmeyerFlask_01/ErlenmeyerFlask_01.usd',
]:
    target = old_candidate / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old / relative, target)
print('Candidate root:', candidate)
PY

/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python \
  /tmp/labware-rebuild-20261022/2026-10-22-labware-integration/prepare_assets.py
```

候选目录的 JSON 会记录候选绝对路径，与原目录的路径文本不同是预期现象；不能因此直接覆盖原验收清单。重建成功后应重新执行官方 `create_actor(convex=True)` 加载、开口落球、动态开盖/倒药负控、功能点和渲染检查，再由单独验收更新标注和 manifest；本配方本身不能证明整机抓取、旋盖或成功率。

## 关键生成逻辑核对

2026-10-22 对当前 `prepare_assets.py` 静态核对：烧杯 `outer_radius_m=0.0348` 表示含倾嘴的可视最大径向范围；`body_outer_radius_m=0.02792` 表示主筒身；碰撞壁配方是 0.02792→0.029 m 的分段近似。源码已有明确说明“lip/spout details are visual only”，重新运行不会退回先前误记的 0.0325 m。当前冻结 interface 不另新增字段；碰撞半径 0.029 m 的复核见 `acceptance_annotation_audit.json`。

当前配方只把 source/spec 的实际 SHA写入新清单，不硬编码旧 source SHA；标注生成与质量、原点、功能点校验由独立 annotation 流程负责。cap 局部原点必须保持顶圆盘中心，z∈[-0.012, +0.002]；不能为了“底面z=0”统一平移瓶盖。
