@echo off
echo ===================================================
echo Starting AI Renewable Energy Monitoring Dashboard...
echo ===================================================

where streamlit >nul 2>nul
if %ERRORLEVEL% equ 0 (
    streamlit run app.py
) else (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m streamlit run app.py
    ) else (
        python -m streamlit run app.py
    )
)
pause
