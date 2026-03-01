@echo off
REM Resolve the folder this script lives in
set SCRIPT_DIR=%~dp0
REM Run Java with classpath including the JSON library and the JAR
java -cp "%SCRIPT_DIR%edit-image.jar;%SCRIPT_DIR%lib\json-20230227.jar" Shell %*
