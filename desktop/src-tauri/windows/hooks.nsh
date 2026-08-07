; Tauri invokes this while Hearth and its bundled backend still exist.
; hearth-backend exports profile identity to %LOCALAPPDATA%\Hearth\retained
; and removes models, packages, conversations, and vector memory. Do not
; abort the uninstaller if a file is locked: setup can recover on reinstall.
!macro NSIS_HOOK_PREUNINSTALL
  IfFileExists "$INSTDIR\resources\backend\hearth-backend.exe" 0 cleanup_done
    nsExec::ExecToLog '"$INSTDIR\resources\backend\hearth-backend.exe" --uninstall-cleanup'
  cleanup_done:
!macroend
