# SUPERFLY for Windows: the script inside SUPERFLY-Setup-Windows.exe.
# Installs into %LOCALAPPDATA%\SUPERFLY (no admin rights needed):
#   the SUPERFLY code, a private Python (via uv), the packages, a desktop
#   shortcut; then starts the Lab, which fetches the fly's connectome and its
#   language model on first run. Later runs go straight to the Lab.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root   = Join-Path $env:LOCALAPPDATA 'SUPERFLY'
$App    = Join-Path $Root 'app'
$Branch = 'claude/trusting-newton-ip3rl0'
$ZipUrl = "https://github.com/gondwanagenesis/fly-brain/archive/refs/heads/$Branch.zip"
$UvUrl  = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
function Say($m) { Write-Host "`n== $m" -ForegroundColor Yellow }
Write-Host ''
Write-Host '   S U P E R F L Y   -  an uplifted fruit fly, still a fly' -ForegroundColor Magenta
Write-Host "   installing into $Root" -ForegroundColor DarkGray
try {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    if ($args -contains '--update' -and (Test-Path $App)) { Remove-Item -Recurse -Force $App }
    if (-not (Test-Path (Join-Path $App 'superfly\quickstart.py'))) {
        Say 'Downloading SUPERFLY'
        $zip = Join-Path $Root 'superfly.zip'
        Invoke-WebRequest $ZipUrl -OutFile $zip
        Say 'Unpacking'
        $tmp = Join-Path $Root 'unpack'
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Expand-Archive $zip -DestinationPath $tmp
        $inner = Get-ChildItem $tmp | Select-Object -First 1
        Move-Item $inner.FullName $App
        Remove-Item -Recurse -Force $tmp
        Remove-Item -Force $zip
    }
    $uv = Join-Path $Root 'uv.exe'
    if (-not (Test-Path $uv)) {
        Say 'Getting a private Python manager (uv)'
        $uz = Join-Path $Root 'uv.zip'
        $ut = Join-Path $Root 'uvtmp'
        Invoke-WebRequest $UvUrl -OutFile $uz
        Expand-Archive $uz -DestinationPath $ut -Force
        Move-Item (Get-ChildItem $ut -Recurse -Filter 'uv.exe' | Select-Object -First 1).FullName $uv -Force
        Remove-Item -Recurse -Force $ut
        Remove-Item -Force $uz
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
    $env:UV_CACHE_DIR = Join-Path $Root 'cache'
    Set-Location $App
    $py = Join-Path $App '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        Say 'Setting up Python 3.12 (private to SUPERFLY)'
        & $uv venv --python 3.12 .venv
        if ($LASTEXITCODE) { throw 'could not create the Python environment' }
    }
    $done = Join-Path $App '.venv\superfly_installed'
    if (-not (Test-Path $done)) {
        Say 'Installing packages (PyTorch and friends, about 1 GB, first run only)'
        & $uv pip install --python $py --index-strategy unsafe-best-match -r requirements-superfly.txt
        if ($LASTEXITCODE) { throw 'package installation failed' }
        'ok' | Out-File $done
    }
    $exe = $env:SUPERFLY_EXE
    $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'SUPERFLY.lnk'
    if ($exe -and (Test-Path $exe) -and -not (Test-Path $lnk)) {
        $home_exe = Join-Path $Root 'SUPERFLY.exe'
        if ($exe -ne $home_exe) { Copy-Item $exe $home_exe -Force }
        $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
        $s.TargetPath = $home_exe
        $s.WorkingDirectory = $Root
        $s.IconLocation = "$home_exe,0"
        $s.Description = 'SUPERFLY - an uplifted fruit fly'
        $s.Save()
        Say 'Added a SUPERFLY shortcut to your desktop'
    }
    Say 'Starting SUPERFLY (the first start also downloads the fly and its language model, ~6.5 GB)'
    $quick = @($args | Where-Object { $_ -ne '--update' })
    & $py -m superfly.quickstart @quick
}
catch {
    Write-Host "`nSomething went wrong: $_" -ForegroundColor Red
    Write-Host 'Please copy the text in this window into an issue at https://github.com/gondwanagenesis/fly-brain/issues'
}
Read-Host "`nPress Enter to close"
