import subprocess

def run_command(cmd, output_file):
    """
    command on terminal
    """
    with open(output_file, "a") as f:

        f.write(cmd + "\n")
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=f, stderr=subprocess.STDOUT)

        process.communicate()


commands = [
    "python train.py --dataset cartoon_data --adv_weight 5  --cycle_weight 200 --iteration 10000  --gpu_ids 0",
    "python train.py --dataset cartoon_data --identity_weight 10  --faceid_weight 1 --iteration 10000  --gpu_ids 0",
    "python train.py --dataset cartoon_data --identity_weight 20  --faceid_weight 1 --iteration 10000  --gpu_ids 0",
    "python train.py --dataset cartoon_data --identity_weight 30  --faceid_weight 1 --iteration 10000  --gpu_ids 0"
]


output_file = 'output_gpu0.txt'
for cmd in commands:
    run_command(cmd, output_file)