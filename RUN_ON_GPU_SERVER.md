# Run trials on the GPU server (216.81.248.162)

Your project is on your **Windows PC**. The GPU is on the **Ubuntu server**. You need to copy the project to the server, then run the script there.

---

## Step 1: Copy the project to the server (run on Windows)

Open a **new** PowerShell terminal **on your Windows machine** (don’t be inside SSH). Run:

```powershell
# Create project folder on server
ssh -i "C:\Users\shivl\Downloads\private_key.pem" ubuntu@216.81.248.162 "mkdir -p ~/stat4830project"

# Copy project files (run from your project folder)
cd "C:\Users\shivl\OneDrive\Desktop\stat4830project folder"
scp -i "C:\Users\shivl\Downloads\private_key.pem" -r configs run_trials.py run_experiment.py requirements.txt src ubuntu@216.81.248.162:~/stat4830project/
```

If `scp` isn’t found, use **Git Bash** or **WSL** and run the same `scp` command there.

---

## Step 2: On the server, install deps and run (inside your SSH session)

In the terminal where you’re already SSH’d in (`ubuntu@0194-dsm2-dla100sxm-prxmx70033:~$`):

```bash
cd ~/stat4830project
pip3 install -r requirements.txt --user
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python3 run_trials.py
```

Use `python3`, not `python`. The script will use the GPU on that machine.

---

## One-liner from Windows (after copying once)

To run the script on the server without opening SSH yourself:

```powershell
ssh -i "C:\Users\shivl\Downloads\private_key.pem" ubuntu@216.81.248.162 "cd ~/stat4830project && python3 run_trials.py"
```

This only works after you’ve copied the project (Step 1) and installed dependencies once (Step 2).
