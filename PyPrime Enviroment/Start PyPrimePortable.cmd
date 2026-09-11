@echo off
cd /d "%~dp0"
if not exist "Notebooks" mkdir "Notebooks"
docker compose up --build
