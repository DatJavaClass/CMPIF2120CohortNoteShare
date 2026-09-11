#!/usr/bin/env bash
## Mac and Linux launcher, run with: bash StartPyPrime.sh
cd "$(dirname "$0")" || exit 1
RAW="https://raw.githubusercontent.com/DatJavaClass/CMPIF2120CohortNoteShare/main/PyPrime%20Environment"

command -v docker >/dev/null || { echo "Docker Desktop is not installed. Get it at https://www.docker.com/products/docker-desktop/ then run this again."; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker Desktop is not running. Open it, wait for the whale to settle, then run this again."; exit 1; }

for f in Dockerfile compose.yaml environment.yml test_environment.py .dockerignore; do
  curl -fsSL -o "$f" "$RAW/$f" && continue
  [ -f "$f" ] && { echo "Could not refresh $f, using the copy already here."; continue; }
  echo "Could not download $f and there is no copy here. Check your internet connection."; exit 1
done
mkdir -p Notebooks
docker compose up --build -d || exit 1

for i in $(seq 30); do curl -s -o /dev/null http://127.0.0.1:8888 && break; sleep 2; done
(open http://localhost:8888 || xdg-open http://localhost:8888) >/dev/null 2>&1
echo; echo "PyPrime is running at http://localhost:8888 and your notebooks save to the Notebooks folder."
read -rsn1 -p "Press any key in this window to stop it."; echo
docker compose down
