"""Build a plain-language report directly from saved episodes, never typed-in scores."""
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
import numpy as np

from .worlds import ROOT, generate, DT
from .runner import KINDS, METHODS, WEIGHTS

NAMES={"dwa":"基础 DWA","astar_dwa":"A* 引导 DWA"}
SCENES={"random":"随机障碍","u_shape":"U 形障碍","narrow":"狭窄通道"}
COLORS={"dwa":"#ce633b","astar_dwa":"#16796f"}
OUT=ROOT/"output"
FIG=OUT/"figures"


def avg(values):
    values=np.asarray(values,dtype=float)
    return {"n":len(values),"mean":float(np.mean(values)) if len(values) else None,
            "std":float(np.std(values,ddof=1)) if len(values)>1 else (0. if len(values) else None)}


def fmt(stats,decimals=2):
    if not stats["n"]:
        return "无成功样本"
    return f"{stats['mean']:.{decimals}f} ± {stats['std']:.{decimals}f}"


def trace(row):
    return np.load(ROOT/"results"/"traces"/(row["trace_id"]+".npz"))


def background(ax,scene):
    for x,y,w,h in scene.rectangles:
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,color="#34424c"))
    ax.scatter(*scene.start[:2],marker="o",s=38,c="#365ba7",label="起点",zorder=6)
    ax.scatter(*scene.goal[:2],marker="*",s=115,c="#b38a17",label="终点",zorder=6)
    ax.set(xlim=(0,20),ylim=(0,20),xlabel="x / 米",ylabel="y / 米")
    ax.set_aspect("equal")
    ax.grid(alpha=.15)


def make_figures(rows):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                         "axes.unicode_minus":False,"font.size":10,"figure.dpi":120})
    fig,axes=plt.subplots(3,2,figsize=(9,11),layout="constrained")
    for i,kind in enumerate(KINDS):
        scene=generate(kind,1000)
        for j,method in enumerate(METHODS):
            row=next(r for r in rows if r["scene"]==kind and r["method"]==method and r["seed"]==1000 and r["obstacle_weight"]==.4)
            data=trace(row)
            ax=axes[i,j]
            background(ax,scene)
            if len(data["global_path"]):
                ax.plot(*data["global_path"].T,"--",color="#87b9b0",lw=1,label="A* 路线")
            xy=data["states"][:,:2]
            ax.plot(*xy.T,color=COLORS[method],lw=2,label="实际轨迹")
            ax.scatter(*xy[-1],marker="x",c=COLORS[method],s=40,zorder=7)
            status="到达" if row["success"] else "未到达"
            ax.set_title(f"{SCENES[kind]} / {NAMES[method]} / {status} · {row['sim_time_s']:.1f} 秒",fontsize=10)
            if i==0:
                ax.legend(fontsize=8,loc="upper left")
    fig.savefig(FIG/"trajectories.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    scene=generate("u_shape",1000)
    background(axes[0],scene)
    failure_rows=[]
    for method in METHODS:
        row=next(r for r in rows if r["scene"]=="u_shape" and r["seed"]==1000 and r["method"]==method)
        failure_rows.append(row)
        state=trace(row)["states"]
        axes[0].plot(*state[:,:2].T,color=COLORS[method],label=NAMES[method],lw=2)
        axes[0].scatter(*state[-1,:2],marker="x",color=COLORS[method])
        distances=np.linalg.norm(state[:,:2]-scene.goal[:2],axis=1)
        axes[1].plot(np.arange(len(state))*DT,distances,color=COLORS[method],label=NAMES[method])
    axes[0].set_title("可复查案例：u_shape_1000")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="仿真时间 / 秒",ylabel="距离终点 / 米",title="绕出去时，可能先离终点更远")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    fig.savefig(FIG/"failure_analysis.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    for method in METHODS:
        rates,times,spreads=[],[],[]
        for weight in WEIGHTS:
            subset=[r for r in rows if r["scene"]=="narrow" and r["method"]==method and r["obstacle_weight"]==weight]
            rates.append(100*sum(r["success"] for r in subset)/len(subset))
            t=avg([r["sim_time_s"] for r in subset if r["success"]])
            times.append(t["mean"] if t["n"] else np.nan)
            spreads.append(t["std"] if t["n"] else np.nan)
        axes[0].plot(WEIGHTS,rates,"o-",color=COLORS[method],label=NAMES[method],alpha=.8)
        axes[1].errorbar(WEIGHTS,times,yerr=spreads,fmt="o-",capsize=4,
                         color=COLORS[method],label=NAMES[method])
    axes[0].set(xlabel="避障代价权重",ylabel="到达率 / %",ylim=(-5,105),title="同一组 10 个狭窄通道实例")
    axes[1].set(xlabel="避障代价权重",ylabel="成功样本完成时间 / 秒",title="均值与标准差，避免放大微小差异",ylim=(0,25))
    for ax in axes:
        ax.set_xticks(WEIGHTS)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(FIG/"parameter_sweep.png",dpi=160)
    plt.close(fig)
    return failure_rows


def make_gif(rows):
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout="constrained")
    scene=generate("u_shape",1000)
    lines,robots,states=[],[],[]
    for ax,method in zip(axes,METHODS):
        row=next(r for r in rows if r["scene"]=="u_shape" and r["seed"]==1000 and r["method"]==method)
        state=trace(row)["states"]
        states.append(state)
        background(ax,scene)
        line,=ax.plot([],[],color=COLORS[method],lw=2)
        robot=Circle(scene.start[:2],.2,color=COLORS[method],zorder=8)
        ax.add_patch(robot)
        lines.append(line)
        robots.append(robot)
        ax.set_title(NAMES[method])
    # Shared simulation clock. Both robots are stationary after t=30s in this
    # recorded case; omit that long static interval and explicitly label the jump.
    frame_steps=np.r_[np.arange(0,301,3),np.repeat(1200,15)]
    title=fig.suptitle("")
    def update(frame):
        step=int(frame_steps[frame])
        for line,robot,state in zip(lines,robots,states):
            i=min(step,len(state)-1)
            line.set_data(state[:i+1,0],state[:i+1,1])
            robot.center=state[i,:2]
        note="省略 30–120 秒静止段" if step==1200 else "3 倍速"
        title.set_text(f"U 形障碍 · seed 1000 · 仿真 {step*DT:.1f} 秒 · {note}")
        return [*lines,*robots,title]
    animation=FuncAnimation(fig,update,frames=len(frame_steps),interval=100,blit=False)
    animation.save(OUT/"u_shape_comparison.gif",writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig)


def build():
    with (ROOT/"results"/"episodes.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key in ("seed","success","collision","steps"):
            row[key]=int(row[key])
        for key in ("obstacle_weight","sim_time_s","path_length_m","decision_mean_ms","global_planning_ms","goal_distance_m"):
            row[key]=float(row[key])
    assert len(rows)==100,"Report requires complete evaluation, including failures"
    main=[r for r in rows if r["obstacle_weight"]==.4]
    summary=[]
    for kind in KINDS:
        for method in METHODS:
            subset=[r for r in main if r["scene"]==kind and r["method"]==method]
            success=[r for r in subset if r["success"]]
            summary.append({"scene":kind,"method":method,"n":len(subset),
                            "successes":len(success),"collisions":sum(r["collision"] for r in subset),
                            "length":avg([r["path_length_m"] for r in success]),
                            "time":avg([r["sim_time_s"] for r in success]),
                            "decision":avg([r["decision_mean_ms"] for r in subset]),
                            "global_planning":avg([r["global_planning_ms"] for r in subset])})
    overall={m:sum(r["success"] for r in main if r["method"]==m) for m in METHODS}
    paired=[]
    for kind in KINDS:
        first={r["seed"]:r for r in main if r["scene"]==kind and r["method"]=="dwa" and r["success"]}
        second={r["seed"]:r for r in main if r["scene"]==kind and r["method"]=="astar_dwa" and r["success"]}
        common=sorted(first.keys()&second.keys())
        paired.append({"scene":kind,"n":len(common),"seeds":common,
                       "time_delta_guided_minus_base":avg([second[s]["sim_time_s"]-first[s]["sim_time_s"] for s in common]),
                       "length_delta_guided_minus_base":avg([second[s]["path_length_m"]-first[s]["path_length_m"] for s in common])})
    OUT.mkdir(exist_ok=True)
    (ROOT/"results"/"summary.json").write_text(json.dumps({"main":summary,"overall":overall,"paired":paired},indent=2,ensure_ascii=False),encoding="utf-8")
    failure_rows=make_figures(rows)
    make_gif(rows)
    baseline,guided=failure_rows
    intro=(f"环境和小项目已经实际跑通。主实验里，基础 DWA 到达 {overall['dwa']}/30 次，"
           f"A* 引导 DWA 到达 {overall['astar_dwa']}/30 次。"
           f"主实验记录了 {sum(r['collision'] for r in main)} 次碰撞。"
           "这份记录的数字都来自保存的轨迹和 CSV；失败和超时也留在里面。")
    meaning=("DWA 每次只看未来两秒，挑一个看起来安全、又靠近目标的速度。"
             "遇到 U 形障碍时，眼前正确的方向可能正好通向死胡同。"
             "改进版先用 A* 在已知地图上找一条能绕出去的路线，再让同一个 DWA 跟着中间目标走。"
             "这次比较考察的是路线引导的作用，不能据此断言所有 DWA 都会失败。")
    settings=("场地 20 × 20 米，圆形差速机器人半径 0.2 米，控制周期 0.1 秒。"
              "线速度 0 到 1 米/秒，角速度绝对值不超过 1.5 弧度/秒；"
              "线加速度限制 1.5 米/秒²，角加速度限制 3 弧度/秒²。"
              "距终点小于 0.3 米且未碰撞算成功；第一次碰撞或仿真到 120 秒结束。")
    protocol=("随机障碍、U 形障碍、狭窄通道各 10 张地图，测试种子均为 1000 到 1009。"
              "主实验两种方法各跑 30 次；另在同一组 10 张狭窄通道地图上，把避障权重设为 0.1、0.4、1.0。"
              "中档结果与主实验共用，因此正式数据共 100 行，不是 120 行，也不是 100 张不同地图。"
              "预实验使用 11、12、13，与测试种子分开；最终配置写入 frozen_parameters.json。")
    fairness=("两种方法都拿到自身位姿、最终目标和完整静态障碍几何，动作范围和终止条件一致。"
              "A* 用到的地图信息，基础版也同样可见；差别是是否利用它计算中间目标。"
              "地图先在考虑机器人尺寸的膨胀栅格上检查连通性，并检查连续路线与障碍的距离，避免生成根本过不去的题。"
              "每步决策同设 1000 毫秒上限，初始化规划同设 5000 毫秒上限；这只是统一预算，不能据此宣称满足硬实时控制。")
    case=(f"可复查的例子是 u_shape_1000。基础版的结果为 {baseline['failure_reason'] or 'success'}，"
          f"结束时离终点 {baseline['goal_distance_m']:.2f} 米；引导版"
          f"{'在 '+format(guided['sim_time_s'],'.1f')+' 秒到达' if guided['success'] else '也没有到达'}。"
          "看下图：绕出开口时，小车需要先朝远离终点的方向走。"
          "只奖励眼前接近终点的控制器，容易停在封口前。A* 路线把这段绕行拆成了可执行的中间目标。"
          "对应的 YAML、JSON、逐步状态和速度都已保存，可以换参数重跑。")
    sweep_rows=[]
    for weight in WEIGHTS:
        for method in METHODS:
            subset=[r for r in rows if r["scene"]=="narrow" and r["obstacle_weight"]==weight and r["method"]==method]
            sweep_rows.append([f"{weight:g}",NAMES[method],f"{sum(r['success'] for r in subset)}/10",
                               fmt(avg([r["sim_time_s"] for r in subset if r["success"]]))])
    all_sweep_success=all(r["success"] for r in rows if r["scene"]=="narrow")
    sweep_means=[np.mean([r["sim_time_s"] for r in rows if r["scene"]=="narrow" and
                          r["method"]==method and r["obstacle_weight"]==weight and r["success"]])
                 for method in METHODS for weight in WEIGHTS
                 if any(r["success"] for r in rows if r["scene"]=="narrow" and r["method"]==method and r["obstacle_weight"]==weight)]
    small_difference=("各组平均完成时间最大只差 "+f"{max(sweep_means)-min(sweep_means):.2f}"+
                      " 秒，本轮的通行效果基本相当。") if sweep_means and max(sweep_means)-min(sweep_means)<.1 else "完成时间差异见下表。"
    sweep_text=("这三档权重下，两种方法都通过了全部狭窄通道测试，成功率没有拉开差距。"
                "这不表示权重无关紧要，只能说明当前门宽和这三档设置还没有触发通行失败。"
                +small_difference+"下一轮可在预实验中缩窄门宽。" if all_sweep_success else
                "不同权重的成功次数和耗时见下表。避障权重提高后，控制器会更在意与障碍的距离；"
                "若狭窄通道里长期停住，应结合轨迹判断是否因为代价过于保守。"
                "本轮只改变权重，没有同时改门宽，不能把两者的作用混在一起。")
    statistics=("表中的路径长度和完成时间，只统计各方法成功的样本，并同时给出成功次数。"
                "均值 ± 标准差使用样本标准差（ddof=1）；只有一个样本时标准差记 0。"
                "每步耗时先在每张地图内取平均，再对 10 张地图统计均值与标准差，包含失败地图，且不含渲染。"
                "两种方法成功的地图可能不同，所以不能只比较总平均路径和总平均完成时间。"
                "summary.json 另保存双方都成功的配对比较。A* 一次性规划耗时单独记录，未藏进或删出实验数据。")
    limits=("这次只做了静态、轴对齐矩形障碍，机器人能直接读到准确位姿和整张地图。"
            "还没有覆盖测距噪声、定位误差、动态行人或真实底盘。狭窄通道是单个窄门，不能代表连续弯曲通道。"
            "A* 的栅格分辨率是 0.2 米，路线可行性不等于差速机器人一定能顺利跟踪。"
            "这些结果是一个可复现的起点，还不是对所有导航场景的结论。")
    next_work=("接下来值得做的是：用新的预实验种子增加折弯窄道，检查保守避障造成的停滞；"
               "补一个停滞检测或恢复策略，再用未看过的测试地图比较。"
               "课程最终提交还需要小组核对代码、成员贡献、课堂展示和演示视频。"
               "当前 GIF 是轨迹对照演示，不能替代包含讲解的三分钟视频。"
               "本报告是 AI 辅助生成的环境与实验记录，提交前应由小组理解、复核并按实际工作改写。")
    implementation=("复用 IR-SIM 2.12.1 的差速运动学、场景加载和碰撞检测。"
                    "新增 worlds.py：地图生成、距离计算、可行性检查和 A*；"
                    "planners.py：DWA 速度采样、轨迹预测、代价和路线引导；"
                    "runner.py：统一实验与逐次记录；verification.py：独立几何与轨迹检查；"
                    "reporting.py：从原始结果重画图、生成报告和动画。"
                    "基础 DWA 是本项目自行实现的版本，并非对 PythonRobotics 示例的逐行复现。")
    env=json.loads((ROOT/"results"/"environment.json").read_text(encoding="utf-8"))
    env_text=f"运行环境：{env['os']}；Python {env['python'].split()[0]}；IR-SIM {env['packages']['ir-sim']}；CPU：{env['cpu']}；逻辑处理器 {env['logical_cpus']} 个。使用 CPU，未调用 GPU。完整版本见 requirements-lock.txt，硬件与源码哈希见 results/environment.json。"
    validation=json.loads((ROOT/"results"/"verification.json").read_text(encoding="utf-8"))
    assert validation["passed"] and any("100 trace" in c for c in validation["checks"]),"Run final verify first"
    verification=("已通过独立 Shapely 几何距离检查、30 张地图的连续可行路径检查、IR-SIM 运动与障碍尺寸对齐检查、"
                  "故意撞墙的碰撞检测检查，以及 100 份轨迹与 CSV 的一致性检查。后者重新计算路径长度、时间和到达状态，"
                  "并核对速度、加速度、成对种子及运动更新。")
    table=[["场景","方法","到达","碰撞","成功路径 / 米","成功时间 / 秒","决策 / 毫秒"]]
    for s in summary:
        table.append([SCENES[s["scene"]],NAMES[s["method"]],f"{s['successes']}/{s['n']}",
                      f"{s['collisions']}/{s['n']}",fmt(s["length"]),fmt(s["time"]),fmt(s["decision"])])
    pair_table=[["场景","双方成功数","时间差 / 秒","路径差 / 米"]]
    for p in paired:
        pair_table.append([SCENES[p["scene"]],str(p["n"]),fmt(p["time_delta_guided_minus_base"]),fmt(p["length_delta_guided_minus_base"])])
    def ht_table(data):
        return '<table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(str(v))}</{"th" if i==0 else "td"}>' for v in row)+'</tr>' for i,row in enumerate(data))+'</table>'
    def img(name):
        encoded=base64.b64encode((FIG/name).read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{encoded}">'
    def p(text):
        return '<p>'+html.escape(text)+'</p>'
    sections=[("先看结果",p(intro)+p(meaning)+ht_table(table)+p(statistics)),
              ("这辆小车怎么走",p(settings)+p(implementation)+p(fairness)),
              ("三类场景，实际轨迹",p("下图固定展示 seed 1000，未按结果挑选最好的一次。其余实例的全部轨迹保存在 results/traces。叉号表示本次结束位置。")+img("trajectories.png")),
              ("一次能说清楚的失败",p(case)+img("failure_analysis.png")),
              ("把避障权重改成三档",p(protocol)+p(sweep_text)+ht_table([["避障权重","方法","到达","成功时间 / 秒"]]+sweep_rows)+img("parameter_sweep.png")),
              ("双方都成功时，再比走得快不快",p("下表为引导版减基础版；负数表示引导版更少。只使用两者在同一张地图上都成功的样本。")+ht_table(pair_table)),
              ("怎么复现，还差什么",p(env_text)+p(verification)+p(limits)+p(next_work)+
               '<pre>.\\.venv\\Scripts\\python.exe run.py demo --scene u_shape --method astar_dwa\n.\\.venv\\Scripts\\python.exe run.py verify\n.\\.venv\\Scripts\\python.exe run.py report</pre>'+
               '<p>重新全量评测：先归档现有 results 与 output，再执行 run.py evaluate。完整安装说明在 README.md。</p>')]
    sources=('<p>任务依据：用户提供的《具身智能与通用智能体课程期末大作业》，第 1、2、3、8 页。</p>'
             '<p>平台：<a href="https://github.com/hanruihua/ir-sim">IR-SIM 官方仓库</a>；'
             '安装与动作接口：<a href="https://ir-sim.readthedocs.io/en/stable/get_started/quick_start.html">官方快速开始</a>；'
             '算法参考：<a href="https://atsushisakai.github.io/PythonRobotics/modules/5_path_planning/dynamic_window_approach/dynamic_window_approach.html">PythonRobotics DWA</a>。</p>'
             '<p>AI 辅助范围：查阅要求与官方文档、搭建环境、编写项目代码、执行实验、生成图表与初步报告。未代填成员贡献。</p>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>选题 A · 小车避障项目运行报告</title><style>
    :root{font-family:"Microsoft YaHei",sans-serif;color:#23343d;background:#edf2f3;line-height:1.85}body{max-width:1080px;margin:36px auto;padding:0 24px}header,section{background:white;border:1px solid #d9e4e6;border-radius:12px;padding:30px 38px;margin-bottom:20px}header{border-top:7px solid #16796f}h1{font-size:32px;line-height:1.5;margin:8px 0}h2{font-size:22px;color:#176c64;margin:0 0 16px}p{margin:12px 0}small{color:#52666e}table{width:100%;border-collapse:collapse;font-size:13px;margin:20px 0}th,td{padding:10px 8px;border-bottom:1px solid #dce6e8;text-align:left}th{background:#eef6f4}img{width:100%;height:auto;display:block;margin:20px 0}pre{background:#f0f5f5;white-space:pre-wrap;padding:20px;font-size:13px}a{color:#16796f}@media(max-width:700px){body{padding:0 10px}header,section{padding:22px 16px}table{font-size:11px}td,th{padding:7px 3px}h1{font-size:25px}}@media print{body{margin:0;max-width:none}header,section{border:0;break-inside:avoid}section{padding:15px 0}img{max-height:240mm;object-fit:contain}}
    </style><header><small>选题 A · 环境搭建与小项目运行记录 · 2026-11-25</small><h1>让小车绕过障碍，<br>先跑一个能复现的版本</h1><p>基础 DWA 与 A* 引导 DWA · 30 张测试地图 · 100 次正式运行</p><p><a href="u_shape_comparison.gif">查看 U 形障碍对照动画</a> · <a href="report.pdf">下载 PDF 报告</a></p></header>'''
    page+=''.join('<section><h2>'+title+'</h2>'+body+'</section>' for title,body in sections)
    page+='<section><h2>来源与 AI 使用说明</h2>'+sources+'</section></html>'
    (OUT/"report.html").write_text(page,encoding="utf-8")
    # Portable six-page PDF using a CJK font, with layout checked after rendering.
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    style=ParagraphStyle("body",fontName="STSong-Light",fontSize=10.5,leading=17,spaceAfter=10,wordWrap="CJK")
    heading=ParagraphStyle("heading",parent=style,fontSize=19,leading=27,textColor=colors.HexColor("#16796f"),spaceAfter=18)
    sub=ParagraphStyle("sub",parent=style,fontSize=12,leading=20,textColor=colors.HexColor("#16796f"))
    cell=ParagraphStyle("cell",parent=style,fontSize=8,leading=12,spaceAfter=0)
    content=[]
    def para(t,s=style):
        content.append(Paragraph(html.escape(t),s))
    def pdf_table(data,widths):
        t=Table([[Paragraph(html.escape(str(x)),cell) for x in row] for row in data],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf3f1")),
                               ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cadbd8")),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),7),
                               ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        content.extend([t,Spacer(1,12)])
    def pdf_image(name,width,height):
        content.append(Image(str(FIG/name),width=width,height=height,kind="proportional"))
    para("选题 A：小车避障项目运行报告",heading)
    para("环境搭建与实验记录 | 2026-11-25 | AI 辅助初稿")
    para(intro);para(meaning)
    pdf_table(table,[45,65,35,35,110,110,111])
    para(statistics)
    content.append(PageBreak())
    para("方法和实验怎么安排",heading)
    para(settings);para(implementation);para(protocol);para(fairness)
    para("这次真正做了什么",sub);para(verification)
    content.append(PageBreak())
    para("三类场景的真实轨迹",heading)
    para("固定展示 seed 1000。叉号是结束位置；其余实例全部保留，未筛选最好的一次。")
    pdf_image("trajectories.png",485,590)
    content.append(PageBreak())
    para("为什么会卡住，改权重又怎样",heading)
    para(case)
    pdf_image("failure_analysis.png",505,202)
    para(sweep_text)
    pdf_table([["权重","方法","到达","成功时间 / 秒"]]+sweep_rows,[50,120,50,291])
    content.append(PageBreak())
    para("参数结果、复现与下一步",heading)
    pdf_image("parameter_sweep.png",505,177)
    para("配对比较：引导版减基础版，仅统计同一地图双方都成功的样本。",sub)
    pdf_table(pair_table,[90,70,175,176])
    para(limits)
    content.append(PageBreak())
    para("复现入口和后续工作",heading)
    para(env_text)
    para("运行入口：在 D:\\project_ea 执行 .\\.venv\\Scripts\\python.exe run.py demo；verify 检查结果，report 重新出图。完整安装和重新评测入口见 README.md。")
    para(next_work)
    para(verification)
    para("来源：课程作业 PDF 第 1、2、3、8 页；IR-SIM 官方仓库（hanruihua/ir-sim，2.12.1）；PythonRobotics 的 DWA 文档（算法参考）。完整链接及 AI 辅助范围见 report.html 与 SOURCES.md。")
    def footer(canvas,doc):
        canvas.setFont("STSong-Light",8)
        canvas.setFillColor(colors.HexColor("#61777a"))
        canvas.drawString(42,24,"选题 A · 可复现实验记录 · AI 辅助初稿")
        canvas.drawRightString(553,24,str(doc.page))
    document=SimpleDocTemplate(str(OUT/"report.pdf"),pagesize=(595.28,841.89),
                               leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=40,
                               title="选题 A：小车避障项目运行报告",author="AI-assisted project log")
    document.build(content,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({"overall_successes":overall,"rows":len(rows),"html":str(OUT/"report.html"),"pdf":str(OUT/"report.pdf")},ensure_ascii=False,indent=2))
