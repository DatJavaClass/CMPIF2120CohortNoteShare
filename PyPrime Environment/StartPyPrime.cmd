@echo off
setlocal
cd /d "%~dp0"
set "RAW=https://raw.githubusercontent.com/DatJavaClass/CMPIF2120CohortNoteShare/main/PyPrime%%20Environment"
set FAIL=

where docker >nul 2>&1 || (echo Docker Desktop is not installed. Get it at https://www.docker.com/products/docker-desktop/ then run this again.& goto :halt)
docker info >nul 2>&1 || (echo Docker Desktop is not running. Open it, wait for the whale to settle, then run this again.& goto :halt)

for %%f in (Dockerfile compose.yaml environment.yml test_environment.py .dockerignore) do call :fetch %%f
if defined FAIL goto :halt
if not exist "Notebooks" mkdir "Notebooks"
docker compose up --build -d || goto :halt

set /a TRIES=0
:wait
curl -s -o nul http://127.0.0.1:8888 && goto :ready
set /a TRIES+=1
if %TRIES% geq 30 (echo Jupyter did not answer in a minute. Run "docker compose logs" here to see why.& goto :halt)
%SystemRoot%\System32\timeout.exe /t 2 /nobreak >nul
goto :wait

:ready
start http://localhost:8888
echo.
echo PyPrime is running at http://localhost:8888 and your notebooks save to the Notebooks folder.
echo Press any key in this window to stop it.
pause >nul
docker compose down
exit /b

:fetch
curl -fsSL -o "%~1" "%RAW%/%~1" && exit /b
if exist "%~1" (echo Could not refresh %~1, using the copy already here.& exit /b)
echo Could not download %~1 and there is no copy here. Check your internet connection.
set FAIL=1
exit /b

:halt
pause
exit /b 1
