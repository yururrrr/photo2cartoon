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
    "python train.py --dataset cartoon_data --adv_weight 3  --cycle_weight 200 --iteration 10000  --gpu_ids 1",
    "python train.py --dataset cartoon_data --identity_weight 10  --faceid_weight 3 --iteration 10000  --gpu_ids 1",
    "python train.py --dataset cartoon_data --identity_weight 20  --faceid_weight 3 --iteration 10000  --gpu_ids 1",
    "python train.py --dataset cartoon_data --identity_weight 10  --faceid_weight 5 --iteration 10000  --gpu_ids 1"
]


output_file = 'output_gpu1.txt'
for cmd in commands:
    run_command(cmd, output_file)