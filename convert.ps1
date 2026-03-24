$ErrorActionPreference = 'Stop'

$InputRoot = "originalData"
$OutputRoot = "convertedData"
$LogFile = "conversion_log.txt"

# Adapter settings (native bioformats output -> notebook-compatible dataset layout)
$AdapterScript = Join-Path $PSScriptRoot "embed_existing_zarr.py"
$AdapterMode = "copy"            # use copy when KeepOnlyAdapted is true
$AdapterScenes = $null            # e.g. "0" to keep only series 0
$AdapterIncludeMacro = $false
$AdapterOverwrite = $true
$KeepOnlyAdapted = $true
$StripLeadingDate = $true         # strips leading YYYYMMDD_ from adapted dataset names

"============================================" | Set-Content $LogFile
" Conversion started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $LogFile
"============================================" | Add-Content $LogFile
"" | Add-Content $LogFile

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Get-ChildItem -Path $InputRoot -Recurse -Filter *.vsi | ForEach-Object {

    $InputFile = $_.FullName
    $BaseName = $_.BaseName
    $ParentName = Split-Path $_.DirectoryName -Leaf
    $Prefix = if ($ParentName.Length -ge 8) { $ParentName.Substring($ParentName.Length - 8) } else { $ParentName }
    $OutDir = Join-Path $OutputRoot ("{0}_{1}.zarr" -f $Prefix, $BaseName)

    Write-Host "Processing $InputFile"
    "Processing $InputFile" | Add-Content $LogFile
    "Output directory: $OutDir" | Add-Content $LogFile

    # --- SKIP if already exists ---
    if (Test-Path -LiteralPath $OutDir) {
        Write-Host "Skipping (already exists): $OutDir"
        "SKIPPED: Output already exists: $OutDir" | Add-Content $LogFile
        "" | Add-Content $LogFile
        return
    }

    # --- Conversion ---
    & conda run -n bf2raw bioformats2raw $InputFile $OutDir `
        --max-workers 16 `
        --tile-width 1024 `
        --tile-height 1024 `
        --compression blosc `
        --compression-properties cname=lz4 2>&1 | Tee-Object -FilePath $LogFile -Append

    $bfRc = $LASTEXITCODE
    "bioformats2raw exit code: $bfRc" | Add-Content $LogFile

    if ($bfRc -ne 0) {
        Write-Host "FAILED (bioformats2raw): $InputFile"
        "FAILED bioformats2raw for $InputFile (exit $bfRc)" | Add-Content $LogFile
        "" | Add-Content $LogFile
        return
    }

    "bioformats2raw SUCCESS" | Add-Content $LogFile

    # --- Repair step ---
    & conda run -n bf2raw python -m microio.cli repair --input $OutDir `
        --persist-table --persist --log-level WARNING 2>&1 | Tee-Object -FilePath $LogFile -Append

    $pyRc = $LASTEXITCODE
    "python exit code: $pyRc" | Add-Content $LogFile

    if ($pyRc -ne 0) {
        Write-Host "FAILED (repair): $InputFile"
        "FAILED python repair for $InputFile (exit $pyRc)" | Add-Content $LogFile
        "" | Add-Content $LogFile
        return
    }

    # --- Adapt native output into notebook-compatible layout ---
    if (-not (Test-Path -LiteralPath $AdapterScript)) {
        Write-Host "WARNING: Adapter script missing: $AdapterScript"
        "ADAPTER WARNING: script not found: $AdapterScript" | Add-Content $LogFile
    }
    else {
        "Starting adapter for $InputFile" | Add-Content $LogFile

        $adapterArgs = @(
            "run", "-n", "bf2raw", "python", $AdapterScript,
            $OutDir,
            "--output-root", $OutputRoot,
            "--source-path", $InputFile,
            "--mode", $AdapterMode
        )
        if ($AdapterScenes) { $adapterArgs += @("--scenes", $AdapterScenes) }
        if ($AdapterIncludeMacro) { $adapterArgs += "--include-macro" }
        if ($AdapterOverwrite) { $adapterArgs += "--overwrite" }
        if (-not $StripLeadingDate) { $adapterArgs += "--keep-leading-date" }

        & conda @adapterArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
        $adapterRc = $LASTEXITCODE
        "adapter exit code: $adapterRc" | Add-Content $LogFile

        if ($adapterRc -ne 0) {
            Write-Host "WARNING: Adapter failed for $InputFile (exit $adapterRc). Keeping native output."
            "ADAPTER WARNING for $InputFile (exit $adapterRc)" | Add-Content $LogFile
        }
        elseif ($KeepOnlyAdapted -and (Test-Path -LiteralPath $OutDir)) {
            Remove-Item -LiteralPath $OutDir -Recurse -Force
            Write-Host "Removed native output after successful adaptation: $OutDir"
            "Removed native output after successful adaptation: $OutDir" | Add-Content $LogFile
        }
    }

    "SUCCESS: $InputFile" | Add-Content $LogFile
    "" | Add-Content $LogFile
}

"============================================" | Add-Content $LogFile
"Conversion finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $LogFile
"============================================" | Add-Content $LogFile

Write-Host "Done. Log written to $LogFile"