; Inno Setup script for BookTrans. Built by .github/workflows/booktrans.yml:
;   iscc /DAppVersion=<version> packaging\installer.iss  (file is UTF-8 with BOM for the Korean names)
#ifndef AppVersion
  #define AppVersion "0.1.2"
#endif

[Setup]
AppId={{6F1C2B0E-6E2A-4C1B-9B53-4A8E2D7C1B10}
AppName=BookTrans
AppVersion={#AppVersion}
AppPublisher=BookTrans contributors
AppCopyright=Copyright (c) 2026 BookTrans contributors
DefaultDirName={autopf}\BookTrans
DefaultGroupName=BookTrans
; Installs per-user without admin rights by default; users may choose all-users.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=BookTrans-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\BookTrans.exe

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\BookTrans\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\BookTrans"; Filename: "{app}\BookTrans.exe"
Name: "{group}\BookTrans 라이선스 및 고지"; Filename: "{app}\THIRD_PARTY_NOTICES.txt"
Name: "{group}\BookTrans 사용 안내"; Filename: "{app}\README.md"
Name: "{group}\{cm:UninstallProgram,BookTrans}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\BookTrans"; Filename: "{app}\BookTrans.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BookTrans.exe"; Description: "{cm:LaunchProgram,BookTrans}"; Flags: nowait postinstall skipifsilent
