---
title: 本机-WPS学校字体环境修复
date: 2026-10-13
updated: 2026-10-13
checked: 2026-10-13
status: 原字体已加载，WPS粗仿宋渲染与用户启动入口修复通过
tags: [环境, WPS, 字体, 开题]
host: admin123-System-Product-Name
---

# 本机-WPS学校字体环境修复

用户要求：学校开题报告的字体与格式不能替换，应修复本机 WPS 的字体环境。用户于 2026-10-13 授权自行寻找并安装字体。原报告见 [[东北大学开题报告初稿-多模态条件控制与场景适应.docx]]，材料入口为 [[01-开题材料清单与证据索引]]。

> [!important] 当前结论与验收边界
> 已安装 18 个原字体文件，并在 WPS 实际进程中确认加载。用户重开后字体提示消失，但粗仿宋文字仍呈黑团；后续通过同一原报告的 FreeType 对照测试恢复显示，并固化用户级启动入口。**实际 WPS 截图中的前 4 页封面、填表说明及加粗子标题已恢复；没有宣称 10 页已在 WPS 逐页核验。** 原报告文件及学校格式未修改。

## 主机与安装位置

| 项目 | 本机核查结果 |
| --- | --- |
| 系统 | Ubuntu 24.04.5 LTS |
| 用户 | admin123，UID 1000 |
| WPS 软件包 | 11.1.0.11733.XA |
| 字体目录 | `/home/admin123/.local/share/fonts/wps-school-fonts` |
| 安装范围 | 当前用户字体、WPS 私有兼容库及两个用户启动入口；系统软件包、系统 FreeType 与学校模板未修改 |
| 核查日期 | 2026-10-13 |

最初本机没有学校要求的黑体、仿宋_GB2312、宋体、华文新魏和 Times New Roman 原字体，fontconfig 会选择其他字体替代。安装 WPS 本身不代表这些字体已经安装。扫描还发现 Steam/Proton 中存在同名字体文件，但其内部名称和版权信息显示为其他字库的替代版本，未采用。

## 安装内容与来源

| 字体名称 | 实际匹配家族 | 文件 |
| --- | --- | --- |
| 黑体 | SimHei／黑体 | `simhei.ttf` |
| 仿宋_GB2312 | FangSong_GB2312／仿宋_GB2312 | `FangSong_GB2312.ttf` |
| 宋体 | SimSun／宋体 | `simsun.ttc`，同时含新宋体 |
| 华文新魏 | STXinwei／华文新魏 | `STXinwei.ttf` |
| Times New Roman | Times New Roman | 常规、粗体、斜体、粗斜体，共 4 个文件 |
| Arial | Arial | 常规、粗体、斜体、粗斜体，共 4 个文件 |
| Symbol | Symbol | `symbol.ttf` |
| Wingdings 系列 | Wingdings、Wingdings 2、Wingdings 3 | 共 3 个文件 |
| Webdings | Webdings | `WEBDINGS.TTF` |
| MT Extra | MT Extra | `mtextra.ttf` |

`FangSong_GB2312` 与现代 `FangSong` 是不同家族，本次安装的是模板指定的前者，未用现代仿宋或 Noto 替代。核对了文件内的家族名称、版本与版权名称表；这是文件身份核查，不是发行方数字签名或授权核验。

- 仿宋_GB2312：来自 [高校工会网站的字体下载页面](https://gonghui.hevttc.edu.cn/info/1152/1514.htm)，内部版权为 GreatWall Computer Co. 1994。
- 华文新魏：来自 [xoofee/chinese_fonts 的 STXINWEI.ttf 文件](https://github.com/xoofee/chinese_fonts/blob/master/STXINWEI.ttf)，内部家族为 STXinwei／华文新魏，版权为 Changzhou SinoType。另以 [Microsoft 字体身份说明](https://learn.microsoft.com/en-us/typography/font-list/stxinwei) 核对名称；该说明不是本次下载源。
- 黑体：来自社区镜像 [chillcicada/ttf-ms-win10-sc-sup](https://github.com/chillcicada/ttf-ms-win10-sc-sup)，固定 commit `f5d2ef2c84e8979b322563a53ea3adb5ab995176`。
- 宋体：来自社区镜像 [streetsamurai00mi/ttf-ms-win10](https://github.com/streetsamurai00mi/ttf-ms-win10)，固定 commit `417eb232e8d037964971ae2690560a7b12e5f0d4`。
- Times New Roman、Arial 和符号字体：来自社区镜像 [dv-anomaly/ttf-wps-fonts](https://github.com/dv-anomaly/ttf-wps-fonts)，固定 commit `8c980c24289cb08e03f72915970ce1bd6767e45a`。

社区镜像不标为 Microsoft 或 WPS 官方分发。字体二进制仅保存在当前用户字体目录，未放入笔记库或 Git；各文件的来源、大小、SHA256、内部家族和安装路径见 [安装核查证据](证据/2026-10-13-WPS学校字体/安装核查.json)。

## 第一阶段：字体安装与只读渲染验证

1. 安装前检查目标目录及符号链接，安装后刷新当前目录的字体缓存。
2. `fc-match` 核对黑体、仿宋_GB2312、宋体、华文新魏、Times New Roman、Arial、Symbol、Wingdings、Wingdings 2、Wingdings 3、Webdings、MT Extra；12 个标签均匹配到上述用户字体目录中的正确家族。
3. 原报告 SHA256 保持为 `5027a0ab095cd0d9ba0c426923f1215c88afd98d15624627dd5695995a02a8a8`；学校模板与原报告未改字体、字号、行距或页面格式。
4. 使用安装后字体，只读渲染原报告为 10 页；全部页面已目视检查，标题、正文、文献、表格和签字区完整。渲染文本覆盖核对 99 个实质段落，没有缺失段落。

该阶段的渲染器是 LibreOffice，不是 WPS；字体匹配通过也不等同于 WPS 实际窗口验收。随后用户要求完整退出，经实际窗口及进程检查确认当时已无 WPS 遗留进程。用户再次打开反馈“不再提示字体，但标题仍缺失或显示异常”，触发下面的实际应用诊断。

## 第二阶段：粗仿宋黑团与 FreeType 对照验证

实际在 WPS 打开原报告后，进程映射显示它已加载黑体、仿宋_GB2312、宋体、华文新魏等本次安装文件。截图中的异常集中在**仿宋_GB2312 加粗**：封面题目与身份字段、“填表说明”以及正文子标题呈黑色墨团；相同字体的普通文字及黑体章节标题正常。因此字体安装已生效，但仍存在渲染问题。

[WPS 社区的原始复现](https://bbs.wps.cn/topic/9157)记录了相同的仿宋／楷体加粗黑团，并报告只给 WPS 使用兼容 FreeType 后恢复；[另一原始故障报告](https://bbs.wps.cn/topic/3137)也反馈合成粗体异常。这些是用户报告，不作为本机成功证据；本次完成了下面的实际对照。

| 对照 | 字体库 | 实际 WPS 结果 |
| --- | --- | --- |
| 原启动方式 | 系统 FreeType 2.13.2，`/usr/lib/x86_64-linux-gnu/libfreetype.so.6.20.1` | 粗仿宋黑团；字体文件已加载 |
| 隔离对照启动 | 本机 Snap 中 FreeType 2.11.1，`/snap/gnome-42-2204/263/usr/lib/x86_64-linux-gnu/libfreetype.so.6.18.1` | 相同原报告的粗仿宋恢复清晰 |
| 修复后的用户桌面入口 | 用户私有 FreeType 2.11.1 | 正常桌面入口启动后，进程映射及实际画面均通过 |

各轮均使用同一个未修改的 DOCX；未更换字体、取消加粗、修改字号或调整行距。对照结果支持当前 WPS 与系统 FreeType 的粗体渲染兼容问题，未定位具体内部函数或提交。

原报告另外存在属性节点顺序兼容风险，但封面异常文字的节点顺序本身正常；此次无需修改 XML 即恢复粗体显示，所以未制作或交付新的结构修复副本。

## 持久修复与实际验收

已将验证成功的库复制到当前用户私有位置，避免依赖 Snap 后续 revision 清理：

`/home/admin123/.local/share/wps-freetype-compat/lib/libfreetype.so.6`

库 SHA256：`ef8908dd08eb7072dd1da355277b069048a86512122592d56b5051796ba62ee2`。系统 `/usr/lib` 与 WPS `/opt` 安装目录均未改动。

仅为以下两个 WPS 启动项增加进程级兼容库加载；两个用户覆盖项原先均不存在，原始桌面配置已备份在私有目录 `backups/`：

- `/home/admin123/.local/share/applications/wps-office-wps.desktop`
- `/home/admin123/.local/share/applications/wps-office-prometheus.desktop`

`desktop-file-validate` 与用户应用索引刷新通过。当前 DOCX 默认入口仍是 `wps-office-prometheus.desktop`，文件关联未改变。实际通过该桌面入口打开原报告，WPS 主进程与文字进程均加载私有兼容库；文字进程同时加载原字体，截图中的封面、填表说明、立论依据 3 个加粗子标题和文献综述 5 个加粗子标题恢复正常，仍显示 10 页。

原 DOCX SHA256 仍为 `5027a0ab095cd0d9ba0c426923f1215c88afd98d15624627dd5695995a02a8a8`，此次未保存或编辑报告。正常显示的原报告已打开在用户 WPS 窗口。

证据：[修复前截图](证据/2026-10-13-WPS学校字体/WPS粗仿宋修复前.png)、[正常桌面入口修复后截图](证据/2026-10-13-WPS学校字体/WPS粗仿宋修复后.png)、[进程及启动项核查](证据/2026-10-13-WPS学校字体/WPS渲染修复核查.json)。验收范围是上述前 4 页的实际画面、加载路径、启动入口和文件哈希，不是 WPS 全功能测试。

修复作用于当前用户的图形启动入口；直接执行 `/usr/bin/wps` 会绕过该入口，需显式加载私有库。后续 WPS 更新或启动方式变化时复核是否仍需此兼容设置。若要撤回，仅移除本次新增的两个用户桌面覆盖项及私有兼容库目录；不要删字体目录或替换系统库。

早期生成的 [[东北大学开题报告初稿-多模态条件控制与场景适应-WPS兼容版.docx]] 替换了字体，已撤回采用建议，只保留为排障历史副本。

操作依据：[WPS 字体安装说明](https://www.wps.cn/learning/question/detail/id/331167)；[WPS Linux 常见问题手册](https://www.wps.cn/kingsoftcommonproblem.pdf)。本记录仅涉及办公软件环境，不改变研究方法或实验状态。
