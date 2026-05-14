#Requires -Version 5.1
<#
.SYNOPSIS
    Clean-rebuilds every configuration of DeepLearning.sln and runs the
    available tests, then prints a summary report.

.DESCRIPTION
    - Discovers MSBuild.exe and vstest.console.exe from the VS installation
      that matches the VisualStudioVersion header in the .sln file (VS 2026 / v18).
    - Rebuilds each Configuration|x64 pair with /t:Rebuild so every translation
      unit is always recompiled (no stale-artifact false-negatives).
    - Runs vstest on every *Test.dll found in the output directory.
      IntelCompiler2022* configurations produce no test projects and are skipped.
    - Emits a summary table at the end.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$SolutionFile = Join-Path $PSScriptRoot "DeepLearning.sln"
$SolutionRoot = Split-Path $SolutionFile -Parent
$Platform     = "x64"

# ---------------------------------------------------------------------------
# Discover VS installation that matches the .sln VisualStudioVersion exactly
# ---------------------------------------------------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found at '$vswhere'. Is Visual Studio installed?"
}

# Extract the major VS version from the .sln header, e.g. 18 from "18.4.11612.150"
$slnMajor = [int](Select-String -Path $SolutionFile -Pattern 'VisualStudioVersion\s*=\s*(\d+)' |
             Select-Object -First 1).Matches[0].Groups[1].Value

# Use an exact major-version band [N, N+1) so a side-by-side VS 2022 (v17)
# or a future VS 2029 (v19) installation cannot be selected accidentally.
$vsBand = "[$slnMajor,$($slnMajor + 1))"
$vsRoot = & $vswhere -version $vsBand -property installationPath -latest 2>$null

if (-not $vsRoot) {
    Write-Error ("No Visual Studio installation with major version $slnMajor found. " +
                 "Expected VS 2026 (v$slnMajor). " +
                 "vswhere returned nothing for -version '$vsBand'")
}

$vsVersion = & $vswhere -version $vsBand -property installationVersion -latest 2>$null
$vsName    = & $vswhere -version $vsBand -property displayName         -latest 2>$null

if ([int]($vsVersion -split '\.')[0] -ne $slnMajor) {
    Write-Error ("VS version mismatch: .sln requires major version $slnMajor " +
                 "(band $vsBand) but vswhere resolved '$vsName' ($vsVersion). Aborting.")
}

$MSBuild = Join-Path $vsRoot "MSBuild\Current\Bin\MSBuild.exe"
$VSTest  = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\TestWindow\vstest.console.exe"

foreach ($tool in $MSBuild, $VSTest) {
    if (-not (Test-Path $tool)) { Write-Error "Required tool not found: '$tool'" }
}

Write-Host ""
Write-Host "VS install : $vsName  ($vsVersion)  ->  $vsRoot"
Write-Host "MSBuild    : $MSBuild"
Write-Host "vstest     : $VSTest"
Write-Host "Solution   : $SolutionFile"
Write-Host ""

# ---------------------------------------------------------------------------
# Configurations to process  (all are x64)
# ---------------------------------------------------------------------------
$Configurations = @(
    "Debug",
    "DebugSingle",
    "Release",
    "ReleaseSingle",
    "IntelCompiler2022",
    "IntelCompiler2022Single"
)

# Configurations for which tests are NOT expected (no test projects in solution)
$NoTestConfigs = @("Debug", "DebugSingle", "IntelCompiler2022", "IntelCompiler2022Single")

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
$Results = [System.Collections.Generic.List[PSCustomObject]]::new()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Get-VsTestCount([string]$LogContent, [string]$Label) {
    if ($LogContent -match "(?m)^\s*${Label}:\s+(\d+)") { return [int]$Matches[1] }
    return 0
}

function Get-VsTestTime([string]$LogContent) {
    if ($LogContent -match "(?m)^Total time:\s+(.+)") { return $Matches[1].Trim() }
    return "-"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
$ConfigCount = $Configurations.Count
$ConfigIndex = 0

foreach ($Cfg in $Configurations) {

    $ConfigIndex++
    $CfgPlat    = "${Cfg}|${Platform}"
    $CfgStart   = Get-Date
    $CfgStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Progress -Activity "DeepLearning build+test" `
                   -Status "[$ConfigIndex/$ConfigCount] $CfgPlat" `
                   -PercentComplete ([int](($ConfigIndex - 1) / $ConfigCount * 100))

    Write-Host ("=" * 70)
    Write-Host ("[$ConfigIndex/$ConfigCount]  $CfgPlat  --  {0}" -f ($CfgStart.ToString("HH:mm:ss")))
    Write-Host ("=" * 70)

    $OutDir   = Join-Path $SolutionRoot "$Platform\$Cfg"
    $BuildLog = Join-Path $SolutionRoot "__build_${Cfg}_${Platform}.log"
    $TestLog  = Join-Path $SolutionRoot "__test_${Cfg}_${Platform}.log"

    # --- Clean output directory ------------------------------------------------
    if (Test-Path $OutDir) {
        Write-Host "  Cleaning $OutDir ..."
        Remove-Item $OutDir -Recurse -Force
    }

    # --- Rebuild ---------------------------------------------------------------
    $BuildSw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host "  >> BUILD  $CfgPlat  (started $(Get-Date -Format 'HH:mm:ss'))"
    Write-Host "  $("- " * 33)"
    $ErrorActionPreference = 'Continue'
    try {
        & $MSBuild $SolutionFile /t:Rebuild /p:Configuration=$Cfg /p:Platform=$Platform /v:minimal /nologo 2>&1 |
            Tee-Object -Variable buildOutput |
            ForEach-Object { Write-Host "  $_" }
        $BuildOk = ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = 'Stop'
    }
    $buildOutput | Out-File $BuildLog -Encoding utf8
    $BuildSw.Stop()

    $WarningLines = @(Select-String -Path $BuildLog -Pattern ': warning ' -CaseSensitive |
                      Select-Object -ExpandProperty Line)
    $WarnCount    = $WarningLines.Count

    $ErrorSummary = ""
    if (-not $BuildOk) {
        $ErrorSummary = (Select-String -Path $BuildLog -Pattern ': error ' |
                         Select-Object -First 3 -ExpandProperty Line) -join "; "
        if (-not $ErrorSummary) { $ErrorSummary = "See $BuildLog" }
    }

    Write-Host "  $("- " * 33)"
    Write-Host ("  << BUILD {0}  warnings: {1}  elapsed: {2}" -f `
                $(if ($BuildOk) {"OK"} else {"FAILED"}), $WarnCount, $BuildSw.Elapsed.ToString("hh\:mm\:ss"))

    # --- Tests -----------------------------------------------------------------
    $TestNotes  = ""
    $Passed     = 0
    $Failed     = 0
    $TotalTests = 0
    $TestTime   = "-"
    $SkipTests  = $false
    $TestDlls   = @()

    if (-not $BuildOk) {
        $SkipTests = $true
        $TestNotes = "Skipped (build failed)"
    } elseif ($NoTestConfigs -contains $Cfg) {
        $SkipTests = $true
        $TestNotes = "No tests (excluded from build)"
    }

    if (-not $SkipTests) {
        $TestDlls = @(Get-ChildItem -Path $OutDir -Filter "*Test.dll" -ErrorAction SilentlyContinue |
                      Select-Object -ExpandProperty FullName)

        if ($TestDlls.Count -eq 0) {
            $TestNotes = "No *Test.dll found in output"
        } else {
            $TestSw = [System.Diagnostics.Stopwatch]::StartNew()
            Write-Host ""
            Write-Host "  >> TESTS  $CfgPlat  (started $(Get-Date -Format 'HH:mm:ss'))  [$($TestDlls.Count) DLL(s)]"
            $TestDlls | ForEach-Object { Write-Host "     $_" }
            Write-Host "  $("- " * 33)"

            $ErrorActionPreference = 'Continue'
            try {
                & $VSTest $TestDlls /logger:console 2>&1 |
                    Tee-Object -Variable testOutput |
                    ForEach-Object { Write-Host "  $_" }
            } finally {
                $ErrorActionPreference = 'Stop'
            }
            $testOutput | Out-File $TestLog -Encoding utf8
            $TestSw.Stop()

            $TestLogContent = Get-Content $TestLog -Raw
            $Passed     = Get-VsTestCount $TestLogContent "Passed"
            $Failed     = Get-VsTestCount $TestLogContent "Failed"
            $TotalTests = Get-VsTestCount $TestLogContent "Total tests"
            $TestTime   = Get-VsTestTime  $TestLogContent

            Write-Host "  $("- " * 33)"
            Write-Host ("  << TESTS  Passed={0}  Failed={1}  Total={2}  vstest-time={3}  elapsed={4}" -f `
                        $Passed, $Failed, $TotalTests, $TestTime, $TestSw.Elapsed.ToString("hh\:mm\:ss"))

            if ($Failed -gt 0) { $TestNotes = "FAILURES present - see $TestLog" }
        }
    }

    $CfgStopwatch.Stop()
    Write-Host ""
    Write-Host ("  Total elapsed for $CfgPlat : {0}" -f $CfgStopwatch.Elapsed.ToString("hh\:mm\:ss"))
    Write-Host ""

    $Results.Add([PSCustomObject]@{
        Configuration = $CfgPlat
        BuildOk       = $BuildOk
        Warnings      = $WarnCount
        Passed        = if ($SkipTests -or $TestDlls.Count -eq 0) { "-" } else { $Passed }
        Failed        = if ($SkipTests -or $TestDlls.Count -eq 0) { "-" } else { $Failed }
        Total         = if ($SkipTests -or $TestDlls.Count -eq 0) { "-" } else { $TotalTests }
        Time          = $TestTime
        Notes         = if ($ErrorSummary) { $ErrorSummary } elseif ($TestNotes) { $TestNotes } else { "-" }
    })
}

Write-Progress -Activity "DeepLearning build+test" -Completed

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"

Write-Host ""
Write-Host ("=" * 70)
Write-Host "## Build & Test Report -- DeepLearning -- $Timestamp"
Write-Host ("=" * 70)
Write-Host ""

$Header  = "| Configuration               | Build  | Warnings | Passed | Failed | Total | Time          | Notes |"
$Divider = "|-----------------------------|--------|----------|--------|--------|-------|---------------|-------|"
Write-Host $Header
Write-Host $Divider

$AllOk = $true
foreach ($r in $Results) {
    $buildMark = if ($r.BuildOk) { "[OK]  " } else { $AllOk = $false; "[FAIL]" }
    $warnMark  = if ($r.Warnings -gt 0) { "[WARN] $($r.Warnings)" } else { "0" }
    $failMark  = if ($r.Failed -ne "-" -and [int]$r.Failed -gt 0) {
                     $AllOk = $false; "[FAIL] $($r.Failed)"
                 } else { $r.Failed }

    $line = "| {0,-27} | {1,-6} | {2,-8} | {3,-6} | {4,-6} | {5,-5} | {6,-13} | {7} |" -f `
            $r.Configuration, $buildMark, $warnMark, $r.Passed, $failMark, $r.Total, $r.Time, $r.Notes
    Write-Host $line
}

Write-Host ""
if ($AllOk) {
    Write-Host "[OK]  READY FOR MASTER - all configurations built and all tests passed."
} else {
    Write-Host "[FAIL] NOT READY - one or more configurations have build failures or test failures."
    $Results | Where-Object { -not $_.BuildOk -or ($_.Failed -ne "-" -and [int]$_.Failed -gt 0) } |
        ForEach-Object { Write-Host "    Blocking: $($_.Configuration) -- $($_.Notes)" }
}
Write-Host ""
Write-Host "Press any key to exit."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
