r"""Run with .venv\Scripts\python.exe run.py --help."""
import argparse


def main():
    parser=argparse.ArgumentParser(description="IR-SIM course A navigation mini-project")
    parser.add_argument("command",choices=["pilot","evaluate","demo","report","verify"])
    parser.add_argument("--scene",choices=["random","u_shape","narrow"],default="u_shape")
    parser.add_argument("--seed",type=int,default=1000)
    parser.add_argument("--method",choices=["dwa","astar_dwa"],default="astar_dwa")
    parser.add_argument("--show",action="store_true",help="Open IR-SIM interactive window for demo")
    args=parser.parse_args()
    if args.command=="report":
        from navlab.reporting import build
        build()
    elif args.command=="verify":
        from navlab.verification import verify
        verify()
    else:
        from navlab import runner
        if args.command=="pilot":
            runner.pilot()
        elif args.command=="evaluate":
            runner.evaluate()
        else:
            runner.demo(args.scene,args.seed,args.method,args.show)


if __name__=="__main__":
    main()
