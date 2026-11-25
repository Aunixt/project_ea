# 选题 A：移动机器人避障导航

这是按课程选题 A 做的可运行起步工程。使用 **IR-SIM 2.12.1** 驱动差速机器人，比较自行实现的基础 DWA 与 A* 路线引导 DWA。

已经在本机搭建独立环境并完成 100 次正式运行：三类场景各 10 个实例、两种方法，以及同一组 10 个狭窄通道实例上的三档避障权重实验。中档结果复用主实验，故总计 100 次。失败和超时均保存。

## 先看什么

- `output/report.html`：说人话的完整报告，图片内嵌，可直接用浏览器打开。
- `output/report.pdf`：排版后的报告。
- `output/u_shape_comparison.gif`：两种方法在同一 U 形障碍里的轨迹回放，统一仿真时钟、3 倍速；30 至 120 秒的静止段明确标注并省略。
- `results/episodes.csv`：100 次正式实验的逐次数据。
- `results/summary.json`：按场景汇总和双方成功实例的配对比较。
- `results/verification.json`：独立检查结果。

报告是环境搭建和实验记录，含 AI 辅助内容。成员贡献需按实际工作填写，小组理解与复核后再组织最终提交材料。

## 当前电脑直接运行

在 PowerShell 中：

```powershell
Set-Location 'D:\project_ea'
.\.venv\Scripts\python.exe run.py demo --scene u_shape --method astar_dwa
.\.venv\Scripts\python.exe run.py demo --scene u_shape --method dwa
.\.venv\Scripts\python.exe run.py verify
```

默认无窗口运行，演示轨迹另存于 `results/demo`，不会修改正式评测 CSV。观看现有动画请打开 `output/u_shape_comparison.gif`。

如果希望打开实时 IR-SIM 窗口：

```powershell
$env:MPLBACKEND = 'TkAgg'
.\.venv\Scripts\python.exe run.py demo --scene random --seed 1000 --method astar_dwa --show
Remove-Item Env:\MPLBACKEND
```

实时窗口需要 Python 带有 Tk。已完成的批量实验和动画生成使用 Agg，不依赖桌面窗口。

## 在新电脑安装

建议使用官方 Windows CPython 3.13；本机使用 `D:\Anaconda\python.exe` 创建独立 `.venv`，不是 PATH 中排在最前的 MSYS Python。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py verify
.\.venv\Scripts\python.exe run.py demo
```

`requirements-lock.txt` 固定全部实际安装版本。最低功能依赖列在 `requirements.txt`。无需 GPU 或模型 API。

## 重新评测与生成报告

```powershell
.\.venv\Scripts\python.exe run.py pilot
```

预实验仅用种子 11、12、13，输出 `results/pilot`。正式测试用种子 1000 至 1009。

`evaluate` 为防止覆盖现有结果，发现 `results/episodes.csv` 会直接报错。重新全量运行时先归档现有产物，例如：

```powershell
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Path "archive\$runStamp" -Force
Move-Item -LiteralPath '.\results' -Destination "archive\$runStamp\results"
Move-Item -LiteralPath '.\output' -Destination "archive\$runStamp\output"
.\.venv\Scripts\python.exe run.py evaluate
.\.venv\Scripts\python.exe run.py verify
.\.venv\Scripts\python.exe run.py report
```

此操作只移动工作区内明确指定的结果目录。源码和场景保留。已跑完时，直接执行 `report` 即可重新生成图表、PDF、HTML 与 GIF；不会重跑控制器。

## 方法和约束

共同输入为机器人真实位姿、最终目标、完整静态障碍几何，均属于题目 A 允许的输入。输出 `[v, omega]`，单位为米/秒、弧度/秒。

场地 20 × 20 米；机器人半径 0.2 米；控制周期 0.1 秒；`0 <= v <= 1`，`abs(omega) <= 1.5`；加速度限制为 1.5 米/秒²和 3 弧度/秒²。小于 0.3 米且未碰撞为成功；碰撞或到 120 秒结束。停滞不会提前截断，运行至同一时间上限再标记 `timeout_stalled`。

DWA 在当前速度和加速度许可的动态窗口内采样，预测未来 2 秒的轨迹。先排除不安全及不满足保守制动净距的轨迹，再按目标距离、朝向、障碍净距、速度和转向代价选取动作。无安全候选时执行动态窗口内的最大制动。A* 引导版本复用完全相同的局部控制器，只将当前追踪目标替换为全局路线上的可见中间目标。

安全候选的代价为 `2.5*d_goal + 0.6*abs(heading_error) + obstacle_weight/(clearance+0.15) + 0.35*(1-v) + 0.05*abs(omega)`。其中 `d_goal` 是两秒预测终点到当前追踪目标的距离，`clearance` 是预测轨迹各采样点的最小机器人边缘净距。权重和所有其他参数在 `Parameters` 中明列。

A* 为 0.2 米栅格、八邻域、禁止斜穿障碍角点。占据栅格考虑机器人半径、额外 0.1 米净距及半格对角线；生成后的折线路径另做连续净距检查。生成所有场景时会检查可达性，但只有引导版将该路线作为控制输入。

每步决策有统一 1000 毫秒硬上限，初始化规划有统一 5000 毫秒上限。该预算并不是 10 Hz 硬实时保证；实测决策耗时见数据。计时剔除渲染，A* 一次规划耗时单列。

## 实验文件

| 文件 | 含义 |
| --- | --- |
| `scenes/*.yaml` | 直接交给 IR-SIM 的场景 |
| `scenes/*.json` | 场景几何、起终点、种子 |
| `results/test_manifest.json` | 正式测试的 30 张地图和可行路径长度 |
| `results/frozen_parameters.json` | 正式评测开始前固定的参数 |
| `results/environment.json` | Python/平台版本、CPU、源码哈希、预算和种子 |
| `results/episodes.csv` | 逐次运行结果和完整参数 |
| `results/traces/*.npz` | 每一步位姿、指令、实际速度、决策耗时和全局路线 |
| `results/pilot/` | 单独的预实验记录，不混入正式结果 |

NPZ 可以用 `numpy.load` 读取，`states` 为 `[x,y,theta]`，包含初始状态；`actions` 和 `applied` 为 `[v,omega]`。`decision_ms` 与每步决策对应。所有坐标和长度以米计、角度以弧度计。

成功样本的长度和完成时间单独统计，并同时报告成功次数。决策耗时包含所有结果。跨地图汇总使用样本标准差；确定性控制器在同一输入下不重复刷成功率。计时仅作为本机参考，未做重复计时实验。

## 代码位置

- `navlab/worlds.py`：三类地图、净距、A*、IR-SIM 场景配置。
- `navlab/planners.py`：自写 DWA、动态窗口、轨迹评价、路线引导。
- `navlab/runner.py`：统一运行、终止判定、计时、CSV 和轨迹记录。
- `navlab/verification.py`：独立几何和仿真接口检查、完整结果核对。
- `navlab/reporting.py`：从原始结果生成图表、动画和报告。
- `SOURCES.md`：开源来源、版本、复用文件和 AI 辅助范围。

目前只覆盖静态矩形地图、准确位姿和已知地图；单个窄门也不能代表所有狭窄环境。尚需成员贡献说明、课堂展示和带讲解的最终演示视频。
