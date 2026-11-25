# 来源、复用与 AI 辅助说明

## 作业依据

用户提供的 `具身智能与通用智能体课程期末大作业.pdf`，重点参考第 1、2、3、8 页。原件位于用户课程文件夹，未更改。文档中的材料清单用来确定项目和评测范围，本次交付定位为可运行的小项目及实验记录。

## IR-SIM

- 官方仓库：https://github.com/hanruihua/ir-sim
- 安装说明：https://ir-sim.readthedocs.io/en/stable/get_started/install.html
- 运行接口：https://ir-sim.readthedocs.io/en/stable/get_started/quick_start.html
- 实际安装：PyPI `ir-sim==2.12.1`，使用发行版本标识，不声称使用某个 Git 提交。
- 安装位置：`.venv/Lib/site-packages/irsim/`。
- 复用：`env/env_base.py` 的环境创建和 `step`；`lib/algorithm/kinematics.py` 的差速运动学；平台的矩形、圆形几何与碰撞状态；平台 YAML 场景加载。
- 未修改第三方包。依赖的许可证保留在原发行包的 dist-info 中，IR-SIM 为 MIT。

## 算法参考

- PythonRobotics DWA：https://atsushisakai.github.io/PythonRobotics/modules/5_path_planning/dynamic_window_approach/dynamic_window_approach.html
- 参考其“在动态速度窗口内预测并评价轨迹”的算法思路，没有复制其源文件。本项目的代价、采样数、制动净距、矢量化实现与路径引导均在 `navlab/planners.py` 明列。
- A* 在 `navlab/worlds.py` 自行实现；没有引入现成导航规划库。

基础 DWA 是本工程选择的一个实现，不等于所有 DWA，也不声称精确复现论文的参数和结果。

## 本项目新增

`run.py`、`navlab/*.py`、场景生成逻辑、评测接口、逐次结果、图表和报告生成逻辑。报告中的数值直接读取本机实验产物，失败和超时均保留。

## AI 辅助范围

本次由 AI 协助读取作业要求与官方文档、创建独立 Python 环境、编写项目、执行预实验和正式实验、验证结果、生成图表、动画与初步报告。用户/小组后续应能解释和修改核心代码，并根据实际工作填写成员贡献。没有虚构四名成员或分工，也没有把 AI 生成的报告标成已经由小组撰写的最终稿。
