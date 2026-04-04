import subprocess
import sys
import time
import os

# Securely setting API credentials for the sub-processes to use natively
os.environ["NUMERAI_PUBLIC_ID"] = "GR6CSQYNJ5LGLFEOGI6ACBXXKZ4VLZ7Q"
os.environ["NUMERAI_SECRET_KEY"] = "ZXJPQ4GI42ACW56X5JOSKXFABLZCXRD7FBC5WXLHZ6SVMFUGAP2CBNQKUJJKCT4Y"


def run_step(command):
    print(f"\n[AUTO-TRAIN] Executing: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Real-time streaming to console
    output_log = ""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end="")
            output_log += line
            
    return process.returncode, output_log

def main():
    print("=== Numerai Autonomous Night-Watchdog started ===")
    
    while True:
        # Run training
        ret_code, log = run_step("python train.py")
        
        if ret_code == 0:
            print("\n[AUTO-TRAIN] Training completely successful! Triggering prediction phase...")
            pred_code, pred_log = run_step("python predict.py --model-name nullsignal21345678 --submit")
            if pred_code == 0:
                print("\n[AUTO-TRAIN] All complete. Predictions uploaded. Sleeping well.")
            else:
                print("\n[AUTO-TRAIN] Prediction failed. Check logs.")
            break
            
        else:
            print(f"\n[AUTO-TRAIN] Process failed with return code {ret_code}")
            
            # Check for missing modules
            if "ModuleNotFoundError" in log or "ImportError" in log:
                print("[AUTO-TRAIN] Detected missing dependency. Attempting auto-install...")
                # Hacky extraction of module name
                import re
                match = re.search(r"No module named '(.+?)'", log)
                if match:
                    lib = match.group(1)
                    print(f"Installing {lib}...")
                    run_step(f"pip install {lib}")
                    print("Retrying training...")
                    continue
                else:
                    print("Could not parse module name. Aborting.")
                    break
                    
            # Check for memory errors
            elif "MemoryError" in log or "Killed" in log or "out of memory" in log.lower():
                print("[AUTO-TRAIN] FATAL: Hit a memory limit. As requested, we will NOT alter the models. Halting.")
                break
                
            else:
                print("[AUTO-TRAIN] Encountered a code bug or unknown error.")
                print("[AUTO-TRAIN] Note: Scripts cannot intelligently rewrite code bugs autonomously without an LLM.")
                print("Halting so it can be reviewed in the morning.")
                break

if __name__ == "__main__":
    main()
