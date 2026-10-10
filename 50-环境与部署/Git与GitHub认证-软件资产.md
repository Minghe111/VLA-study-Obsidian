---
title: Git与GitHub认证-软件资产
aliases: [Git登录方式, GitHub登录方式, Git软件资产]
tags: [软件资产, Git, GitHub, 环境, RoboTwin]
created: 2026-09-23
updated: 2026-10-11
status: 两私库历史上传已验证；Windows代码已同步，笔记按时间线合并；服务器状态按日期单独记录
---

# Git 与 GitHub 认证：软件资产

关联：[[环境与部署索引]] · [[服务器172.17.27.166-robotwin]] · [[服务器-172.17.27.166]]。

用途：保存项目代码、维护自己的私有仓库，并从官方仓库获取更新。本笔记记录工具、认证方式、恢复步骤与核验结果；不保存密码、Token、私钥或一次性登录码。

## 已核验的软件与身份

| 项目 | 2026-09-23 核验结果 |
| --- | --- |
| GitHub 账号 | `Minghe111` |
| 本机系统账号 | `admin123` |
| GitHub CLI | `v2.101.0`，安装在 `/home/admin123/.local/bin/gh` |
| CLI 来源 | GitHub 官方 `cli/cli` Release；下载文件的 SHA256 已与 Release 元数据核对 |
| Git LFS | 本机已有 `git-lfs/3.4.1`，本次模型文件不需要 LFS |
| Git 传输协议 | HTTPS |
| 登录方式 | `gh auth login` 浏览器设备授权，用户本人已完成授权 |
| 凭据存放位置 | 系统密钥环（`gh auth status` 返回 `keyring`），未保存到本笔记或项目文件 |
| Git 凭据助手 | `/home/admin123/.local/bin/gh auth git-credential` |
| CLI OAuth 权限 | 本次客户端报告 `gist`、`read:org`、`repo`、`workflow`；不是仅限两个仓库的权限 |
| 本机认证验收 | `gh auth status` 确认账号；已能读取两个私有仓库的 Git 远程 |

Git 提交署名与 GitHub 登录是两回事：本项目提交署名使用 `Minghe111 <liminghe@neu.edu.cn>`；实际上传权限来自上述 GitHub 认证。

## 私有仓库资产

| 仓库 | 用途 | 可见性 |
| --- | --- | --- |
| [RoboTwin-RealMan](https://github.com/Minghe111/RoboTwin-RealMan) | RoboTwin、RM65-B、因时四代触觉手及仿真接入 | Private，已通过 GitHub API 核验 |
| [XPolicyLab-RealMan](https://github.com/Minghe111/XPolicyLab-RealMan) | 独立的策略子模块及项目修改 | Private，已通过 GitHub API 核验 |

两个仓库均属于 `Minghe111`。Obsidian 仓库和 `lingbot-vla-v2` 不属于本次迁移范围。

## 新机器登录或重新登录

安装官方 GitHub CLI 后，在需要上传代码的机器上运行。下面使用本机实际安装路径；其他机器改成它自己的 `gh` 路径。

```bash
/home/admin123/.local/bin/gh auth login --hostname github.com --git-protocol https --web --skip-ssh-key
/home/admin123/.local/bin/gh auth setup-git --hostname github.com
/home/admin123/.local/bin/gh auth status
```

按照提示打开 <https://github.com/login/device>，输入本次显示的一次性代码，确认登录账号为 `Minghe111` 并授权 GitHub CLI。一次性代码会过期，每次重新发起登录都使用新代码。

本次选择允许 Git 使用 GitHub CLI 凭据；实际配置的 HTTPS 凭据助手已经验证。GitHub CLI 通常使用系统凭据存储；其他机器若没有密钥环，应检查 `gh auth status` 报告的存储方式，不假定仍与本机相同。[官方登录说明](https://cli.github.com/manual/gh_auth_login)

用实际私有仓库验证认证，不仅看登录提示：

```bash
git ls-remote https://github.com/Minghe111/RoboTwin-RealMan.git
git ls-remote https://github.com/Minghe111/XPolicyLab-RealMan.git
```

仓库完全为空时，成功读取可能没有输出，应同时检查退出码。

## 下载和上传目标保护

```bash
git clone --recurse-submodules https://github.com/Minghe111/RoboTwin-RealMan.git
cd RoboTwin-RealMan
python3 scripts/configure_private_remote.py
git remote -v
git -C XPolicyLab remote -v
```

- `origin`：自己的私有仓库，日常下载、上传目标。
- `upstream`：官方仓库，获取更新；其推送地址设置为禁用值。
- 默认上传目标为 `origin`。上传前检查只允许对应的 `Minghe111` 私有仓库，同时暂时阻止超过 1 GB 的普通文件或 LFS 文件。
- Git hooks 属于本地配置。每次新克隆需运行配置脚本；该脚本也配置已下载的 XPolicyLab 子模块。
- 这是防误操作措施，可以被人为绕过，不等同于 GitHub 服务端权限隔离。

修改子模块时，先在 XPolicyLab 内提交、上传，再回到主仓库提交子模块的新版本指针。主仓库不会自动保存子模块中未提交的文件。

## 本机与 SSH166 的区别

本机已有可用的 GitHub HTTPS 认证。SSH166 指 `liminghe@172.17.27.166`，不等同于 GitHub 身份认证；现有本机到服务器的 SSH 密钥也不能据此认为已获 GitHub 授权。

本次检查 SSH166 直接访问 GitHub 443 超时，尚未在服务器安装个人 GitHub 凭据。上传由本机完成；服务器代码已使用经过提交核对的 Git bundle 同步。以后若需要服务器独立 `git pull/push`，应先解决 GitHub 网络访问，再在服务器单独配置认证。不要把本机已登录写成服务器也已登录。

## 大文件规则

用户要求：单文件超过 1 GB 暂不上传 GitHub，后续放到 Hugging Face。本次保守采用 `1,000,000,000` 字节作为边界。需要 LFS 的合格文件可另行跟踪；安装 Git LFS 不会自动把所有文件转换为 LFS。

本次四代手模型约 40 MiB，最大单文件约 4.31 MiB，直接保存到 Git。官方 `objects.zip` 为 3,737,778,549 字节，`background_texture.zip` 为 10,970,687,027 字节，保留原位置；未完成下载缓存不作为完整资产上传。具体清单随项目保存在 `LARGE_FILES_PENDING.md`。

## 常见检查与恢复

| 现象 | 检查方式 |
| --- | --- |
| `could not read Username` | 检查该机器的 `gh auth status`，再运行 `gh auth setup-git --hostname github.com` |
| 本机可以、服务器不行 | 分别检查两台机器的网络与身份认证，不能继承结论 |
| SSH `Permission denied (publickey)` | 表示现用 SSH 密钥没有通过认证；本次已选择经过验证的 HTTPS 方式 |
| 上传被本项目 hook 拦截 | 检查 `git remote -v` 和目标是否为自己的仓库，不直接关闭检查 |
| 重新克隆后没有上传保护 | 运行仓库的 `scripts/configure_private_remote.py` |

退出本机 CLI 登录可使用 `gh auth logout --hostname github.com --user Minghe111`。若要撤销 GitHub 侧授权，在 GitHub Settings → Applications → Authorized OAuth Apps 中管理 GitHub CLI；本机退出与服务端撤销是不同操作。[GitHub CLI 退出说明](https://cli.github.com/manual/gh_auth_logout)

## 本次迁移验收

2026-09-23 已完成真实上传，并从 GitHub 重新克隆主仓库及私人子模块验收。

| 仓库 | GitHub `main` 最终提交 | 本机、SSH166 |
| --- | --- | --- |
| RoboTwin-RealMan | [`5628656`](https://github.com/Minghe111/RoboTwin-RealMan/commit/56286567103cf58bfa6191ed889908c5c3a0baa3) | 均为同一提交，`main` 跟踪 `origin/main`，工作区干净 |
| XPolicyLab-RealMan | [`9700776`](https://github.com/Minghe111/XPolicyLab-RealMan/commit/9700776207411cd0abe37a60ee5b59aeaf527e83) | 均为同一提交，子模块指针一致，工作区干净 |

- 上传范围：两端代码修改、项目配置、四代手及旧参考模型、SDK 定义来源、组件验证材料和复现说明。两个仓库均保留官方历史，提交说明使用中文。
- 新仓库原有的初始化提交 `7c3223c` 已通过合并保留，没有强制覆盖。根目录保留官方 MIT 许可证；初始化时的 Apache 模板原样保存在 `migration_records/initial_repository_LICENSE.txt`。
- 私有状态由已登录 CLI 再次查询确认，两仓库均为 `private=true`。
- 新克隆的主仓库与私人子模块 Git 对象连接性检查通过，93 个四代手整机网格齐全；3 组结构和控制合同测试在新克隆及 SSH166 均通过。
- 本机迁移副本在现有 RoboTwin 环境和官方物体资产下完成组件仿真，`passed=true`。这不代表完整 benchmark、VLA/RL 或真机验收；全新机器仍需安装环境和官方物体资产。
- 本机、SSH166 均已配置私有 `origin`、默认推送目标和官方 `upstream` 推送禁用。上传前目标检查通过；没有向官方仓库推送。
- 原工作目录内官方资产包自带 LFS 属性，曾导致 164 个模型文件显示虚假修改。实际文件哈希与上传版本一致，现已在两个 RealMan 模型目录添加局部属性覆盖；没有修改官方资产目录规则，也没有把这些小文件转成 LFS。
- 大于 1 GB 的文件仍留在原位置，未上传 GitHub 或 Hugging Face。

工作目录：

- 本机：`/home/admin123/liminghe/vla-benchmark/RoboTwin`。
- SSH166：`/bigdata2/liminghe/VLA-benchmark/RoboTwin`。

回退资料：

- 本机：`/home/admin123/liminghe/vla-benchmark/migration_backups/20260923-private-repositories/`，含原 Git 历史、未提交修改、新文件、服务器模型快照、上传版本 bundle、新克隆验收目录和最终核对记录。
- SSH166：`/bigdata2/liminghe/VLA-benchmark/migration_backups/20260923-private-repositories/`，含迁移前历史 bundle、原文件快照及迁移版本 bundle。
- 两端需要暂存的原修改另以“迁移私人仓库前备份 2026-09-23”命名的 Git stash 保留，未自动删除。恢复前应先查看内容，不要直接覆盖现在的主分支。

## 2026-10-03 Windows RoboTwin 远端切换

用户要求检查当前 Windows RoboTwin 是否仍连接官方仓库，是则切换到自己的 RealMan 私有仓库。本节针对 `E:/Workspace/VLA-benchmark/RoboTwin`；前文 9 月 23 日的“本机”为 Linux `/home/admin123/...`，不能视为 Windows 此前已经完成相同配置。

本次实测原 `origin` 的 fetch/push 均为 `https://github.com/RoboTwin-Platform/RoboTwin.git`。GitHub API 核实 `Minghe111/RoboTwin-RealMan` 为 private、默认分支 main，当前授权可读取与推送；`git ls-remote` 及随后 fetch 均成功。

已完成的本地配置：

| 项目 | 结果 |
| --- | --- |
| origin fetch / push | `https://github.com/Minghe111/RoboTwin-RealMan.git` |
| upstream fetch | `https://github.com/RoboTwin-Platform/RoboTwin.git` |
| upstream push | `DISABLED`，沿用既有防误推约定 |
| main 跟踪 | `origin/main`，已获取私有仓库的远端引用 |
| 默认推送目标 | `remote.pushDefault=origin`、`branch.main.pushRemote=origin` |

操作前后 HEAD 和全部未提交文件的 Git 状态一致。本地 HEAD 保持 `ea8b211`；本次读取的私有远端 main 为 `c826f28d3bb499d7bb8e5afe94236a5b26d49b2f`，提交关系为本地独有 12 个、远端独有 5 个。此次只配置远端和读取引用，没有合并、切换代码版本、提交或推送；不能写成 Windows 代码已与私有 main 同步。

上一步仅切换 RoboTwin 主仓库，当时 `XPolicyLab` 子模块与 `.gitmodules` 尚为官方地址；用户随后要求一并切换子模块，结果见下节。主仓库未提交的 reference 适配和四代手辅助文件继续保留。

证据类型：本机 Git 实测与 GitHub 仓库元数据；核查日期 2026-10-03。关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B接入RoboTwin仿真]]。

### 同日追加：XPolicyLab 子模块切换

用户追加要求切换 XPolicyLab。GitHub API 确认 `Minghe111/XPolicyLab-RealMan` 为 private、默认分支 main，当前授权可读取与推送；`ls-remote` 和 fetch 成功。

- 子模块 `origin` 的 fetch/push 改为 `https://github.com/Minghe111/XPolicyLab-RealMan.git`。
- 官方 `https://github.com/XPolicyLab/XPolicyLab.git` 保留为 `upstream`，其 pushurl 为 `DISABLED`；默认推送 origin。
- 主仓库 `.gitmodules` 的 `submodule.XPolicyLab.url` 同步改为上述私有地址，并执行 `git submodule sync -- XPolicyLab` 更新本地登记。
- 现有子模块 main 分支跟踪 `origin/main`；没有切换当前检出状态，仍 detached HEAD 于 `fa431ec`。

操作前后主仓库 HEAD、索引中的子模块 gitlink、子模块 HEAD、未提交文件状态以及 `utils/robot/_robot_info.json` SHA-256 均保持一致。本次读取的私有子模块 main 为 `3f3804c1405965df1259abfa135e7c626b0f209f`；当前检出提交落后它 4 个提交，现有本地 main 分支落后它 2 个提交。

此次没有执行 checkout、submodule update、合并、提交或推送。`.gitmodules` 地址变更保留为主仓库未提交改动；远端配置完成不等于子模块代码已经更新到私有 main。

## 2026-10-03 Windows 私有代码合并

在上面远端切换完成后，用户继续要求“开始合并代码”。本节记录当前 Windows 工作目录的实际合并结果；前两节“未合并”是此前阶段的历史状态，不再代表当前工作树。关联：[[RM65-B接入RoboTwin仿真]] · [[服务器172.17.27.166-robotwin]]。

| 仓库 | 合并结果 |
| --- | --- |
| RoboTwin | `main` 创建合并提交 `2c8ba401aa51fb13296f9b6e1d6bb19487523cf6`（合并私有 RealMan 适配与本地 RoboTwin 更新）；双亲分别为原本地 `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` 和私有远端 `c826f28d3bb499d7bb8e5afe94236a5b26d49b2f`，保留双方历史 |
| XPolicyLab | 从 detached HEAD 切回已有 `main`，快进到私有 `3f3804c1405965df1259abfa135e7c626b0f209f`；该提交包含原检出 `fa431ec`，父仓库 gitlink 同步为 `3f3804c` |
| 远端与同步状态 | 两仓库继续使用私有 HTTPS origin、官方 upstream 且禁用 upstream 推送；本次没有推送。RoboTwin 比已获取的 origin/main 领先 13 个提交，XPolicyLab main 与 origin/main 一致 |

### 本地修改的保留与取舍

- 私有 `c826f28` 已主动删除旧 `integrations/realman_rm65b/` 和 reference 配置，改用 Gen4 原生 24 维接口；本次保留新版入口，两侧维度表均为双臂 `[6,6]`、双手 `[6,6]`，不重新注册旧 14 维 reference。
- 合并前所有未提交文件均有逐文件 SHA-256 快照、补丁、stash 和历史备份分支。旧 reference 的 18 个历史文件及两份 YAML 与私有早期 `8e40c74` 一致；旧 README 只缺历史版本提示。它们保存在备份中，不重新放回当前正式入口。
- 本地独有 `verify_control.py` 保存在快照和主仓库 stash 内。它依赖旧 `control_sim/smoke_task` 以及固定 14 维协议，尚未迁入 Gen4，不能当成 24 维控制验证器使用。
- 独有 `integrations/realman_rm65b_gen4/hand_coupling.py` 已按原 SHA-256 恢复到工作树，保持未跟踪、未提交、未启用；其 tendon/follower 驱动策略与当前控制器逐步写入方式仍需单独协调验证。

回退与证据目录：`E:/Workspace/VLA-benchmark/deployment/git_merge_20261003_175908/`，含 `manifest.json`、两仓库原文件快照、工作树和索引补丁、测试输出与 `merge-validation.json`。

| 保护对象 | 可恢复引用 |
| --- | --- |
| 两仓库原 HEAD | 各自 `backup/pre-merge-20261003-175908` 分支 |
| RoboTwin 原未提交修改 | stash 对象 `76244aec5ce3bb6ac062c3fb938d3d92a54e7c69` |
| XPolicyLab 原未提交修改 | stash 对象 `08aa72d1921c193ddf07bdcd5de5bac7ab18f304` |

stash 均保留，没有自动 pop 或删除。需要找回旧文件时先查看快照或单独提取，避免整包恢复导致旧 reference 注册覆盖 Gen4。

### 本次验证及边界

- 自动合并无文本冲突；复核保留官方的 clutter 边界修复、采集 seed 记录、评测独立随机流，同时保留私有 RealMan 初始化、逐步控制、24 维观测和多通道动作转换。
- 在实际合并后的 Windows 工作树运行现有合同测试，3 组全部通过：模型连通性和网格、独立通道及联动、厂家惯性和几何。93 个 STL 齐全且不是 LFS 指针；两个仓库的 Gen4 维度表一致。
- 33 个相关 Python 文件 AST 解析通过（包含仅保留的 hand_coupling 草稿），5 份 YAML 可解析，子模块版本及私有 URL 一致，Git 空白检查通过。独立审查还验证了动作转换的 1/6 通道往返、默认动作回填、非法输入拒绝和指令随机流隔离。
- Windows 首次默认 GBK 读取 UTF-8 模型清单失败；使用以下 UTF-8 启动参数重跑通过，没有据此改动模型文件或算法。

```powershell
python -X utf8 -B integrations/realman_rm65b_gen4/test_contract.py assets/embodiments/realman-rm65b-inspire-gen4 -v
```

本机 Python 环境缺少 SAPIEN，本次未重新运行完整原生仿真、训练或闭环任务评测，也未更新 166 服务器。合并后的主仓库除上述 hand_coupling 草稿外无额外未提交改动，子仓库工作树干净。证据类型：2026-10-03 本地 Git 状态、提交历史对比与实际结构/接口测试。

### 同日追加：合并提交已推送私有仓库

用户随后明确要求“推送”。2026-10-03 已重新 fetch 核对两仓库无远端新增分叉，使用现有 `.githooks/pre-push` 检查通过后，先处理 XPolicyLab，再推送 RoboTwin；没有强制推送，也没有上传未提交草稿。

| 私有仓库 | GitHub main 核验结果 |
| --- | --- |
| [RoboTwin-RealMan](https://github.com/Minghe111/RoboTwin-RealMan) | 从 `c826f28` 快进到 `2c8ba401aa51fb13296f9b6e1d6bb19487523cf6`，推送成功；包含此前待上传的 13 个提交 |
| [XPolicyLab-RealMan](https://github.com/Minghe111/XPolicyLab-RealMan) | `3f3804c1405965df1259abfa135e7c626b0f209f` 已在远端，推送返回 up to date |

推送后通过 `git ls-remote origin refs/heads/main` 重新查询，两仓库远端 main 均与本地 HEAD 一致。主仓库 gitlink 仍指向 `3f3804c`；`hand_coupling.py` 保持本地未跟踪草稿，备份和 stash 均保留。

Windows 未设置 `core.hooksPath`；本次用实际 Python 路径显式运行仓库现有检查脚本，再执行普通 push，没有改写检查脚本或持久配置。上传及远端核验记录：`E:/Workspace/VLA-benchmark/deployment/git_merge_20261003_175908/push-verification.json`。本次只同步 GitHub 代码，未更新服务器或重跑仿真。

## 2026-10-10 两私库提交与复核（Linux本机）

本节本机为 Linux `/home/admin123/liminghe/vla-benchmark`；Windows 的后续同步单独记录。

本机已再次验证 Minghe111 账号的 HTTPS 密钥环认证，并通过 GitHub API 确认两个目标均为 Private。XPolicyLab `main` 上传 [`5680076`](https://github.com/Minghe111/XPolicyLab-RealMan/commit/5680076094c2c3fd5f4f3c9ada5d19d113fcb196)，RoboTwin `main` 上传 [`931ce21`](https://github.com/Minghe111/RoboTwin-RealMan/commit/931ce21f4f0e63feab661f349634d5a25bd92681)，主仓库引用同一子模块版本；两个工程本机工作区干净。

此处只复核本机 GitHub 登录和上传，没有重新验证服务器 GitHub 认证或同步其工作区。9 月迁移表保留为历史快照。本次新增范围及组件证据见 [[2026-10-09-RealMan从161同步与组件验收]]；凭据不写入笔记或 Git，仓库外实验目录不包含在这次提交中。

## 2026-10-11 Windows本机按时间线合并

用户在前次核查后明确授权编辑commit并上传，以时间线合并。本节本机为Windows `E:/Workspace/VLA-benchmark`及`E:/Workspace/VLA-study-Obsidian`；10月9–10日的Linux本机实验仍保留原主机语境。

- 笔记本地基线`01a8485`，本次刷新远程为`227fc49`（含10月9–10日记录）；先保存本地14项改动为`6e90f52`，再合并两侧历史。四处冲突涉及框架、Git资产、环境索引和指南；保留两侧正文，按记录日期衔接采集、训练与推理结果。
- 实验入口为[[RM65-B接入RoboTwin仿真#项目进度时间线（2026-10-11合并）]]。10月8日“尚未推理”是历史快照，10月10日40000检查点已有8条样例；追加6条中2条抓起并保持，仍不代表完整SR。最新训练证据截至10月10日20:47，本次未重新核验服务器。
- Windows RoboTwin已快进至`931ce21f4f0e63feab661f349634d5a25bd92681`，XPolicyLab至`5680076094c2c3fd5f4f3c9ada5d19d113fcb196`，子模块指针一致、工作区干净。原两个未跟踪适配脚本与远程blob相同，已先备份再由已上传版本接管，未丢弃独有代码。
- 合并前14项笔记／视频及两个脚本已逐文件SHA256备份至`E:/Workspace/VLA-benchmark/deployment/git_timeline_merge_20261011_003301/`。该备份与仓库外数据、模型、Conda、实验缓存不纳入笔记提交；凭据不写入Git。
- 按用户授权提交并推送笔记`main`，保留双方已有提交历史，不强制覆盖远程；上传结果以本次任务的远程分支读回为准。
