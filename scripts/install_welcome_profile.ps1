$ErrorActionPreference = "Stop"

# Resolve this clone's root dynamically from the script location
$repoRootResolved = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath

if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}

$existing = ""
if (Test-Path $profilePath) {
    $existing = Get-Content $profilePath -Raw
}

# Strip any prior TXCAT hook to avoid duplication or broken snippets
$clean = [regex]::Replace($existing, "# TXCAT welcome hook.*?# End TXCAT welcome hook\s*", "", "Singleline")

# If earlier errors or welcome text were appended to the profile, drop them
if ($clean -match "DotSourceNotSupported" -or $clean -match "Cannot dot-source this command" -or $clean -match "Welcome to TXCAT") {
    $clean = ""
} else {
    # Remove any stray welcome line while preserving other profile content
    $clean = [regex]::Replace($clean, "Welcome to TXCAT.*(?:\r?\n)?", "", "IgnoreCase")
}

# Build the hook with the resolved path baked in (avoid format braces)
$template = @'
# TXCAT welcome hook (auto-generated)
try {
    $repoRoot = '__ROOT__'
    if ((Get-Location).Path -like "$repoRoot*") {
        if (-not $global:TXCAT_WELCOME_SHOWN) {
            $global:TXCAT_WELCOME_SHOWN = $true
            Write-Host "Welcome to TXCAT. Run scripts\welcome.ps1 for the full guide." -ForegroundColor Cyan
        }
    }
} catch {
    Write-Warning "TXCAT welcome hook failed: $_"
}
# End TXCAT welcome hook
'@

$snippet = $template.Replace('__ROOT__', $repoRootResolved)
$newContent = ($clean.TrimEnd() + "`n" + $snippet).TrimStart("`n")

Set-Content -Path $profilePath -Value $newContent -Encoding UTF8
Write-Host "Installed minimal TXCAT welcome hook to $profilePath (root: $repoRootResolved)"
