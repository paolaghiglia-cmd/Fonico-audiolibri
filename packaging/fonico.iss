; Installer per Windows (Inno Setup)
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6C7E2B7A-3F0E-4E55-9C1B-FA61C0A7D2E1}
AppName=Fonico audiolibri
AppVersion={#AppVersion}
AppPublisher=Fonico audiolibri
DefaultDirName={autopf}\Fonico audiolibri
DefaultGroupName=Fonico audiolibri
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Fonico-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
UninstallDisplayIcon={app}\Fonico.exe
ChangesAssociations=yes

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul desktop"; GroupDescription: "Icone:"

[Files]
Source: "..\dist\Fonico\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Fonico audiolibri"; Filename: "{app}\Fonico.exe"
Name: "{autodesktop}\Fonico audiolibri"; Filename: "{app}\Fonico.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Classes\.fonico"; ValueType: string; ValueName: ""; ValueData: "FonicoAudiolibro"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\FonicoAudiolibro"; ValueType: string; ValueName: ""; ValueData: "Audiolibro Fonico"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\FonicoAudiolibro\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Fonico.exe"" ""%1"""

[Run]
Filename: "{app}\Fonico.exe"; Description: "Apri Fonico audiolibri"; Flags: nowait postinstall skipifsilent
