---
title: Git与GitHub认证-软件资产
aliases: [Git登录方式, GitHub登录方式, Git软件资产]
tags: [软件资产, Git, GitHub, 环境, RoboTwin]
created: 2026-09-23
updated: 2026-09-23
status: 本机认证与两仓库上传已验证，两端代码已同步
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
