"""
Convert legacy .xls files (BIFF/OLE compound document) to .xlsx via Excel COM.

Why this exists
---------------
Python 3.14 currently has no working ``xlrd`` wheel (the canonical reader for
.xls files is unmaintained and xlrd >= 2.0 dropped .xls support). On Windows
machines where Excel is installed, the most reliable workaround is to ask
Excel itself to convert the file.

Usage
-----
    python src/data/xls_to_xlsx.py path/to/file.xls

Output
------
Writes ``<file>.xlsx`` next to the input (or under ``src/data/converted/`` if
you pass ``--outdir``). Excel is invoked headless (no GUI popups).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

# Make sure we're on Windows
if os.name != "nt":
    raise SystemExit("xls_to_xlsx.py requires Windows + Excel COM.")


def _run_powershell(xls_paths: Iterable[str], outdir: str | None) -> None:
    """Drive Excel via PowerShell COM."""
    paths_block = "\n".join(f'    "{p}",' for p in xls_paths)
    outdir_block = f'"{outdir}"' if outdir else "$null"
    script = f"""
$ErrorActionPreference = 'Stop'
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$paths = @({paths_block.rstrip(',')})
$outdir = {outdir_block}
foreach ($xls in $paths) {{
    $xlsAbs = (Resolve-Path $xls).Path
    if ($outdir) {{
        if (-not (Test-Path $outdir)) {{ New-Item -ItemType Directory -Path $outdir | Out-Null }}
        $out = Join-Path $outdir ([System.IO.Path]::GetFileNameWithoutExtension($xlsAbs) + '.xlsx')
    }} else {{
        $out = [System.IO.Path]::ChangeExtension($xlsAbs, '.xlsx')
    }}
    Write-Host "Converting $xlsAbs -> $out"
    $wb = $excel.Workbooks.Open($xlsAbs)
    $wb.SaveAs($out, 51)   # 51 = xlOpenXMLWorkbook
    $wb.Close($false)
}}
$excel.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
"""
    # Run via powershell
    import subprocess
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Excel conversion failed.")
    print(result.stdout)


def main():
    ap = argparse.ArgumentParser(description="Convert .xls -> .xlsx via Excel COM.")
    ap.add_argument("xls", nargs="+", help="One or more .xls files.")
    ap.add_argument("--outdir", default=None,
                    help="If set, write .xlsx files into this directory.")
    args = ap.parse_args()

    for p in args.xls:
        if not p.lower().endswith(".xls"):
            raise SystemExit(f"Not an .xls file: {p}")

    _run_powershell(args.xls, args.outdir)
    print("[DONE]")


if __name__ == "__main__":
    main()
