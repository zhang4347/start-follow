@echo off
rem Grab-room mode: keeps trying the table number set in the launch settings file.
rem "start" goes through ShellExecute so the UAC (admin) prompt works.
start "" /D "%~dp0" "%~dp0StarFollow.exe" --grab
