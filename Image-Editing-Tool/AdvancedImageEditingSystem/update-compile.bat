@echo off
setlocal enabledelayedexpansion

echo Cleaning out old build…
rmdir /s /q out
mkdir out

echo Finding all Java sources…
dir /b /s src\*.java > sources.txt

echo Compiling with org.json on the classpath…
javac -cp "lib/json-20230227.jar" -d out @sources.txt
if errorlevel 1 exit /b 1

echo Writing MANIFEST.MF…
echo Main-Class: Shell> out\MANIFEST.MF

echo Packaging into edit-image.jar…
jar cfm edit-image.jar out\MANIFEST.MF -C out .

echo Removing temporary files…
del sources.txt

echo ✅ Build complete.
