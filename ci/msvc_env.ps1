# SPDX-License-Identifier: MIT
# Loads the MSVC x64 developer environment (cl, link, Windows SDK, VS's CMake and Ninja)
# into this PowerShell session; under GitHub Actions also into the following steps.
#
#   . ci/msvc_env.ps1
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio with the C++ x64 tools found" }
$vars = cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set"
foreach ($line in $vars) {
    if ($line -match "^(.*?)=(.*)$") {
        Set-Item "env:$($matches[1])" $matches[2]
        if ($env:GITHUB_ENV) { "$($matches[1])=$($matches[2])" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV }
    }
}
"MSVC environment from $vs"
