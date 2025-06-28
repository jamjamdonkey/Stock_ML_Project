@echo off
cd /d %~dp0
cd ..
echo ===============================
echo KOSPI 종목 추천 시스템 실행 중...
echo ===============================

python app\app.py

pause
