"""Build the double-click installers in dist/ (run from the repo root):

    python installer/build_installers.py

    dist/SUPERFLY-Setup-Windows.exe  a small console program (cross-compiled
                                     with `python -m ziglang cc`) that runs
                                     installer/superfly_setup.ps1
    dist/SUPERFLY-Mac.zip            SUPERFLY.app, which opens Terminal and
                                     runs installer/superfly-install.sh
    dist/superfly.ico / .icns        icons rendered from docs/img/superfly_logo.html

Rendering the icon needs node + playwright; without them the previous icon
files in dist/ are reused.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INST = ROOT / "installer"
DIST = ROOT / "dist"
SCRIPT_URL = ("https://raw.githubusercontent.com/gondwanagenesis/fly-brain/"
              "claude/trusting-newton-ip3rl0/installer/superfly-install.sh")

RENDER_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 600, height: 760 } });
  await p.goto('file://' + src);
  await p.waitForTimeout(1500);
  await p.evaluate(() => {
    const s = document.getElementById('logo');
    s.setAttribute('viewBox', '14 14 572 572');
    s.setAttribute('width', '256'); s.setAttribute('height', '256');
  });
  const el = await p.$('#logo');
  await el.screenshot({ path: out, omitBackground: true });
  await b.close();
})();
"""

C_MAIN = r"""
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include "script.h"

/* SUPERFLY for Windows: writes the installer script to %TEMP% and runs it
 * with PowerShell (present on every Windows 10/11). Source: installer/. */
int main(int argc, char **argv)
{
    char exe[MAX_PATH], tmp[MAX_PATH], cmd[8192];
    SetConsoleTitleA("SUPERFLY");
    GetModuleFileNameA(NULL, exe, MAX_PATH);
    SetEnvironmentVariableA("SUPERFLY_EXE", exe);
    if (!GetTempPathA(MAX_PATH, tmp)) { printf("No temp folder.\n"); getchar(); return 1; }
    strncat(tmp, "superfly_setup.ps1", MAX_PATH - strlen(tmp) - 1);
    FILE *f = fopen(tmp, "wb");
    if (!f) { printf("Could not write %s\n", tmp); getchar(); return 1; }
    fwrite("\xEF\xBB\xBF", 1, 3, f);
    fwrite(SCRIPT, 1, sizeof(SCRIPT) - 1, f);
    fclose(f);
    snprintf(cmd, sizeof cmd,
             "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"%s\"", tmp);
    for (int i = 1; i < argc; i++) {
        strncat(cmd, " ", sizeof cmd - strlen(cmd) - 1);
        strncat(cmd, argv[i], sizeof cmd - strlen(cmd) - 1);
    }
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        printf("Could not start PowerShell (error %lu).\n", GetLastError());
        getchar();
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}
"""

MAC_LAUNCHER = f"""#!/bin/bash
# SUPERFLY.app: opens Terminal and runs the SUPERFLY installer / launcher.
T="${{TMPDIR:-/tmp}}/superfly-launch.command"
cat > "$T" <<'EOS'
#!/bin/bash
if [ -f "$HOME/.superfly/app/installer/superfly-install.sh" ]; then
  bash "$HOME/.superfly/app/installer/superfly-install.sh"
else
  curl -fsSL {SCRIPT_URL} | bash
fi
EOS
chmod +x "$T"
open -a Terminal "$T"
"""

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>SUPERFLY</string>
  <key>CFBundleDisplayName</key><string>SUPERFLY</string>
  <key>CFBundleIdentifier</key><string>org.superfly.launcher</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>SUPERFLY</string>
  <key>CFBundleIconFile</key><string>superfly</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
"""


def icons():
    png = DIST / "superfly_256.png"
    js = INST / "_render_icon.js"
    js.write_text(RENDER_JS)
    try:
        subprocess.run(["node", str(js), str(ROOT / "docs/img/superfly_logo.html"), str(png)],
                       check=True, cwd=INST)
    except Exception as e:                       # keep the previous icons
        print("icon render skipped:", e)
    finally:
        js.unlink(missing_ok=True)
    if png.exists():
        data = png.read_bytes()
        ico = struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(data), 22) + data
        (DIST / "superfly.ico").write_bytes(ico)
        entry = b"ic08" + struct.pack(">I", 8 + len(data)) + data
        (DIST / "superfly.icns").write_bytes(b"icns" + struct.pack(">I", 8 + len(entry)) + entry)


def windows_exe():
    ps = (INST / "superfly_setup.ps1").read_text()
    assert ps.isascii(), "keep the PowerShell script ASCII"
    lines = ['static const char SCRIPT[] =']
    for line in ps.splitlines():
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{esc}\\r\\n"')
    lines[-1] += ";"
    build = DIST / "_build"
    build.mkdir(exist_ok=True)
    (build / "script.h").write_text("\n".join(lines) + "\n")
    (build / "main.c").write_text(C_MAIN)
    (build / "app.rc").write_text('1 ICON "superfly.ico"\n')
    (build / "superfly.ico").write_bytes((DIST / "superfly.ico").read_bytes())
    out = DIST / "SUPERFLY-Setup-Windows.exe"
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-windows-gnu", "-O2",
           "-o", str(out), "main.c", "app.rc"]
    r = subprocess.run(cmd, cwd=build, capture_output=True, text=True)
    if r.returncode:                             # without the icon resource
        print("with icon failed, building without:", r.stderr[-400:])
        subprocess.run(cmd[:-1], cwd=build, check=True)
    for p in build.iterdir():
        p.unlink()
    build.rmdir()
    for p in DIST.glob("SUPERFLY-Setup-Windows.pdb"):
        p.unlink()
    print("built", out, out.stat().st_size, "bytes")


def mac_zip():
    out = DIST / "SUPERFLY-Mac.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def add(name, data, mode):
            zi = zipfile.ZipInfo(name)
            zi.external_attr = (mode & 0xFFFF) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)
        for d in ("SUPERFLY.app/", "SUPERFLY.app/Contents/", "SUPERFLY.app/Contents/MacOS/",
                  "SUPERFLY.app/Contents/Resources/"):
            add(d, b"", 0o40755)
        add("SUPERFLY.app/Contents/Info.plist", INFO_PLIST, 0o100644)
        add("SUPERFLY.app/Contents/MacOS/SUPERFLY", MAC_LAUNCHER, 0o100755)
        if (DIST / "superfly.icns").exists():
            add("SUPERFLY.app/Contents/Resources/superfly.icns", (DIST / "superfly.icns").read_bytes(), 0o100644)
    print("built", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    DIST.mkdir(exist_ok=True)
    icons()
    windows_exe()
    mac_zip()
