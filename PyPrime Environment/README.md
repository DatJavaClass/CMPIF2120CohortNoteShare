# PyPrime Environment

Python installs are where semesters go to die. One person has Python 3.9, one has three copies and none of them can see pandas. PyPrime ends that. It is the class Python setup in a box: Python 3.14, pandas, NumPy, scikit learn, XGBoost, matplotlib, seaborn and JupyterLab, the same versions on every machine. You download one file, double click it, and JupyterLab opens in your browser.

## What Docker is, in one paragraph

Docker Desktop runs a small sealed Linux computer inside your computer. Everything PyPrime needs lives in that box, so it cannot fight with anything already on your machine. Your notebooks do not live in the box. They live in a normal folder next to the start file, so if the box is ever deleted your work is still right there.

## Setup

1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and reboot if it asks. On Windows it will want WSL 2, say yes. Open it once and wait for the whale icon to sit still.
2. Make a folder you will find again, say Documents\PyPrime.
3. Download one file from this folder and put it there. Windows: `StartPyPrime.cmd`. Mac or Linux: `StartPyPrime.sh`. On GitHub, click the file name, then the download button on the right (the arrow pointing down into a tray).
4. Windows: double click the file. If Windows says it protected your PC, click More info, then Run anyway. Windows says that about anything downloaded outside its store. Mac or Linux: open Terminal, type `bash ` with a space after it, drag the file onto the Terminal window, press Enter.
5. Wait. The first run builds the box, a few minutes on good internet, and a black window scrolls through all of it. When it finishes your browser opens JupyterLab at http://localhost:8888. Every later start takes seconds.
6. Save your work in the `Notebooks` folder that appeared next to the start file. It is the only folder the box can see.
7. Done for the day? Press any key in the black window and the box shuts down. Closing the window with the X leaves it running until you quit Docker Desktop, which is harmless.

The start file downloads the recipe fresh each time it runs, so when the package list changes here, you get it on your next start without doing anything.

## What is in the box

| Package | Version |
|---|---|
| Python | 3.14 |
| pandas | 3.0.5 |
| NumPy | 2.5.3 |
| scikit learn | 1.9.1 |
| XGBoost | 3.4.1 (CPU) |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| JupyterLab | 4.6.3 |

The other files here are the recipe. `Dockerfile` and `compose.yaml` tell Docker how to build and run the box, `environment.yml` is the package list, and `test_environment.py` fails the build if a package is missing. You do not download them, the start file does.

## No Docker?

Already have conda? Skip the box entirely. Download `environment.yml`, run `conda env create -f environment.yml`, then `conda activate PyPrimePortable`. Same packages, same versions.

I am not a Docker expert, nor will I ever claim to be. This was assembled with AI help, then run start to finish on my own machine before it went up here. If it breaks on yours, tell me which step and we will fix it.
