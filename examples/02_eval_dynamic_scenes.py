import argparse
import sys
import omnigibson as og
from realm.eval import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    # parser.add_argument('--perturbation_id',nargs='+',type=int,default=[0])
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--model', type=str, required=True, default=None)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--save_frames', type=int, default=1)
    parser.add_argument('--save_video', type=int, default=1)

    args = parser.parse_args()
    assert args.model is not None
    assert args.experiment_name is not None
    log_dir = args.log_dir if args.log_dir is not None else "/app/logs"
    log_dir += f"/{args.experiment_name}"
    log_dir += f"/{args.model}"
    log_dir += f"/{args.run_id}" if args.run_id is not None else ""

    evaluate(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        model=args.model,
        port=args.port,
        log_dir=log_dir,
        save_frames=args.save_frames,
        save_video=args.save_video
    )
    og.shutdown()
    sys.exit(0)
