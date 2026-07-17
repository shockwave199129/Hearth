; Hearth NSIS Installer Script
; Customized installer for Windows

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

; General Settings
Name "Hearth ${VERSION}"
OutFile "Hearth-${VERSION}-installer.exe"
InstallDir "$PROGRAMFILES\Hearth"
InstallDirRegKey HKCU "Software\Hearth" ""
RequestExecutionLevel user

; MUI Settings
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; Installer Sections
Section "Install"
  SetOutPath "$INSTDIR"
  
  ; Copy installer wizard
  File /r "installer\*.*"
  
  ; Copy main app
  File /r "app\*.*"
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  
  ; Create registry entries
  WriteRegStr HKCU "Software\Hearth" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Hearth" "DisplayName" "Hearth"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Hearth" "UninstallString" "$INSTDIR\Uninstall.exe"
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\Hearth"
  CreateShortcut "$SMPROGRAMS\Hearth\Hearth.lnk" "$INSTDIR\hearth-installer.exe"
  CreateShortcut "$SMPROGRAMS\Hearth\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\Hearth.lnk" "$INSTDIR\hearth-installer.exe"
SectionEnd

Section "Uninstall"
  ; Remove files
  RMDir /r "$INSTDIR"
  
  ; Remove shortcuts
  RMDir /r "$SMPROGRAMS\Hearth"
  Delete "$DESKTOP\Hearth.lnk"
  
  ; Remove registry entries
  DeleteRegKey HKCU "Software\Hearth"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\Hearth"
SectionEnd
