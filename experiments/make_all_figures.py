import subprocess
import sys

def main():
    subprocess.check_call([sys.executable, "experiments/sweep_snr.py"])

if __name__ == "__main__":
    main()
