param(
    [switch]$SkipExport,
    [string]$ProjectFile = "",
    [string]$RawDataFile = "",
    [string]$SegmentationFile = "",
    [string]$RoiStart = "",
    [string]$RoiStop = ""
)

$ErrorActionPreference = "Stop"

$IlastikExe = "C:\Program Files\ilastik-1.4.2-gpu\ilastik.exe"
$IlastikPython = "C:\Program Files\ilastik-1.4.2-gpu\python.exe"
$OutputDataset = "exported_data"
$PoreLabel = 2

if ([string]::IsNullOrWhiteSpace($ProjectFile)) {
    $ProjectFile = Join-Path $PSScriptRoot "4-class.ilp"
}
$ProjectFile = $ProjectFile.Trim('"')
$ProjectDir = Split-Path -Parent $ProjectFile

if ([string]::IsNullOrWhiteSpace($RawDataFile)) {
    $RawDataFile = Join-Path $ProjectDir "2-o89-300.tiff"
}
$RawDataFile = $RawDataFile.Trim('"')

if (-not [string]::IsNullOrWhiteSpace($SegmentationFile)) {
    $SegmentationFile = $SegmentationFile.Trim('"')
}

function Assert-FileExists {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Name not found: $Path"
    }
}

function Convert-RoiToIlastikCutout {
    param([string]$Start, [string]$Stop)

    if ([string]::IsNullOrWhiteSpace($Start) -and [string]::IsNullOrWhiteSpace($Stop)) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($Start) -or [string]::IsNullOrWhiteSpace($Stop)) {
        throw "Provide both -RoiStart and -RoiStop as z,y,x. Example: -RoiStart 140,0,0 -RoiStop 160,300,300"
    }

    $startParts = $Start.Split(",") | ForEach-Object { [int]($_.Trim()) }
    $stopParts = $Stop.Split(",") | ForEach-Object { [int]($_.Trim()) }
    if ($startParts.Count -ne 3 -or $stopParts.Count -ne 3) {
        throw "ROI must use z,y,x. Example: -RoiStart 140,0,0 -RoiStop 160,300,300"
    }

    return "[(0,0,$($startParts[0]),$($startParts[1]),$($startParts[2])),(1,1,$($stopParts[0]),$($stopParts[1]),$($stopParts[2]))]"
}

function Copy-FileShared {
    param([string]$Source, [string]$Destination)

    $sourceStream = [System.IO.File]::Open(
        $Source,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite
    )
    try {
        $destinationStream = [System.IO.File]::Open(
            $Destination,
            [System.IO.FileMode]::Create,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $sourceStream.CopyTo($destinationStream, 1048576)
        }
        finally {
            $destinationStream.Dispose()
        }
    }
    finally {
        $sourceStream.Dispose()
    }
}

function Find-SegmentationExport {
    param([string]$ProjectPath, [string]$ExplicitPath)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        return $ExplicitPath
    }

    $dir = Split-Path -Parent $ProjectPath
    $preferred = Join-Path $dir "5-seged.h5"
    if (Test-Path -LiteralPath $preferred) {
        return $preferred
    }

    $patterns = @("*.h5", "*.hdf5", "*.tif", "*.tiff")
    $candidates = foreach ($pattern in $patterns) {
        Get-ChildItem -LiteralPath $dir -Filter $pattern -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match "(?i)(seg|segment|simple|label)" -and
                $_.Name -notmatch "(?i)(preclas|project|class\.ilp)"
            }
    }

    $candidate = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($candidate) {
        return $candidate.FullName
    }

    throw "No segmentation export found next to project: $dir. Export Simple Segmentation first, or pass -SegmentationFile."
}

Assert-FileExists -Path $IlastikExe -Name "ilastik.exe"
Assert-FileExists -Path $IlastikPython -Name "ilastik python.exe"
Assert-FileExists -Path $ProjectFile -Name "ilastik project"
Assert-FileExists -Path (Join-Path $PSScriptRoot "count_porosity.py") -Name "count_porosity.py"

$env:PYTHONIOENCODING = "utf-8"
$env:HDF5_USE_FILE_LOCKING = "FALSE"

$CountingInputFile = Find-SegmentationExport -ProjectPath $ProjectFile -ExplicitPath $SegmentationFile

if (-not $SkipExport) {
    Assert-FileExists -Path $RawDataFile -Name "raw data"

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ilastik_porosity_{0:yyyyMMdd_HHmmss}_{1}" -f (Get-Date), $PID)
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

    $projectCopy = Join-Path $tempRoot ([System.IO.Path]::GetFileName($ProjectFile))
    $outputFile = Join-Path $tempRoot "simple_segmentation.h5"
    $logFile = Join-Path $tempRoot "ilastik.log"

    Write-Host "Copying project to temp:"
    Write-Host "  $projectCopy"
    try {
        Copy-FileShared -Source $ProjectFile -Destination $projectCopy
    }
    catch {
        Write-Host ""
        Write-Host "Cannot copy the ilastik project while it is locked by the ilastik GUI."
        Write-Host "Save the project first. If it is still open, Windows may still block headless access."
        Write-Host "You can count the last exported segmentation without re-exporting:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -SkipExport"
        Write-Host ""
        exit 3
    }

    Write-Host "Temp output:"
    Write-Host "  $outputFile"

    $ilastikArgs = @(
        "--headless",
        "--readonly",
        "--project=$projectCopy",
        "--logfile=$logFile",
        "--export_source=Simple Segmentation",
        "--output_filename_format=$outputFile",
        "--output_internal_path=$OutputDataset",
        "--input_axes=zyx",
        $RawDataFile
    )

    $cutout = Convert-RoiToIlastikCutout -Start $RoiStart -Stop $RoiStop
    if ($cutout) {
        $ilastikArgs += "--cutout_subregion=$cutout"
        Write-Host "Exporting Simple Segmentation ROI: $cutout"
    } else {
        Write-Host "Exporting full Simple Segmentation..."
    }

    & $IlastikExe @ilastikArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ilastik export failed with exit code $LASTEXITCODE"
    }

    $CountingInputFile = $outputFile
} else {
    Assert-FileExists -Path $CountingInputFile -Name "existing segmentation export"
}

Assert-FileExists -Path $CountingInputFile -Name "segmentation export"

Write-Host ""
Write-Host "Counting porosity..."
& $IlastikPython (Join-Path $PSScriptRoot "count_porosity.py") `
    --input $CountingInputFile `
    --dataset $OutputDataset `
    --pore-label $PoreLabel

if ($LASTEXITCODE -ne 0) {
    throw "porosity counting failed with exit code $LASTEXITCODE"
}
