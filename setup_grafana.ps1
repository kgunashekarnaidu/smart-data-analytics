# ============================================================================
# setup_grafana.ps1 — Install and configure Grafana for DataML Pro
# Run this script as Administrator in PowerShell
# ============================================================================
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$grafanaVersion = "11.1.0"
$installerUrl = "https://dl.grafana.com/oss/release/grafana-$grafanaVersion.windows-amd64.msi"
$installerPath = "$env:TEMP\grafana-$grafanaVersion.msi"
$grafanaPath = "C:\Program Files\GrafanaLabs\grafana"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  DataML Pro — Grafana Setup                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Download Grafana ---
if (Test-Path $installerPath) {
    Write-Host "[✓] Installer already downloaded." -ForegroundColor Green
} else {
    Write-Host "[1/5] Downloading Grafana $grafanaVersion..." -ForegroundColor Yellow
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "[✓] Downloaded successfully." -ForegroundColor Green
    } catch {
        Write-Host "[✗] Download failed: $_" -ForegroundColor Red
        Write-Host "    Download manually from: https://grafana.com/grafana/download?platform=windows" -ForegroundColor Yellow
        exit 1
    }
}

# --- Step 2: Install Grafana ---
Write-Host "[2/5] Installing Grafana..." -ForegroundColor Yellow
$process = Start-Process -FilePath "msiexec.exe" `
    -ArgumentList "/i `"$installerPath`" /quiet /qn" `
    -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host "[✓] Grafana installed." -ForegroundColor Green
} elseif ($process.ExitCode -eq 1638) {
    Write-Host "[✓] Grafana already installed (existing version detected)." -ForegroundColor Green
} else {
    Write-Host "[✗] Installation failed (exit code: $($process.ExitCode))." -ForegroundColor Red
    exit 1
}

# --- Step 3: Configure Grafana (custom.ini) ---
Write-Host "[3/5] Configuring Grafana for iframe embedding..." -ForegroundColor Yellow

$customIni = "$grafanaPath\conf\custom.ini"
$customIniContent = @"
; DataML Pro — Grafana custom configuration
; This file overrides defaults.ini settings

[security]
allow_embedding = true
cookie_samesite = disabled

[auth.anonymous]
enabled = true
org_role = Viewer

[server]
http_port = 3000
"@

Set-Content -Path $customIni -Value $customIniContent -Encoding UTF8 -Force
Write-Host "[✓] custom.ini created at: $customIni" -ForegroundColor Green

# --- Step 4: Copy provisioning files ---
Write-Host "[4/5] Copying provisioning and dashboard files..." -ForegroundColor Yellow

# Datasources provisioning
$dsSource = "$appDir\grafana\provisioning\datasources"
$dsDest = "$grafanaPath\conf\provisioning\datasources"
if (-Not (Test-Path $dsDest)) { New-Item -ItemType Directory -Path $dsDest -Force | Out-Null }
Copy-Item -Path "$dsSource\*" -Destination $dsDest -Force
Write-Host "    → Datasource configs copied" -ForegroundColor Gray

# Dashboard provider provisioning
$dbProvSource = "$appDir\grafana\provisioning\dashboards"
$dbProvDest = "$grafanaPath\conf\provisioning\dashboards"
if (-Not (Test-Path $dbProvDest)) { New-Item -ItemType Directory -Path $dbProvDest -Force | Out-Null }
Copy-Item -Path "$dbProvSource\*" -Destination $dbProvDest -Force
Write-Host "    → Dashboard provider configs copied" -ForegroundColor Gray

# Dashboard JSON files
$dashSource = "$appDir\grafana\dashboards"
$dashDest = "$grafanaPath\public\dashboards\dataml"
if (-Not (Test-Path $dashDest)) { New-Item -ItemType Directory -Path $dashDest -Force | Out-Null }
Copy-Item -Path "$dashSource\*" -Destination $dashDest -Recurse -Force
Write-Host "    → Dashboard JSON files copied" -ForegroundColor Gray

Write-Host "[✓] All provisioning files in place." -ForegroundColor Green

# --- Step 5: Start Grafana service ---
Write-Host "[5/5] Starting Grafana service..." -ForegroundColor Yellow

try {
    $svc = Get-Service -Name "Grafana" -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -eq "Running") {
            Restart-Service -Name "Grafana" -Force
            Write-Host "[✓] Grafana service restarted." -ForegroundColor Green
        } else {
            Start-Service -Name "Grafana"
            Write-Host "[✓] Grafana service started." -ForegroundColor Green
        }
    } else {
        Write-Host "[!] Grafana service not found. Starting manually..." -ForegroundColor Yellow
        Start-Process -FilePath "$grafanaPath\bin\grafana-server.exe" `
            -WorkingDirectory "$grafanaPath" -WindowStyle Hidden
        Write-Host "[✓] Grafana server started manually." -ForegroundColor Green
    }
} catch {
    Write-Host "[!] Could not auto-start Grafana: $_" -ForegroundColor Yellow
    Write-Host "    Start manually: $grafanaPath\bin\grafana-server.exe" -ForegroundColor Yellow
}

# --- Done ---
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ Grafana Setup Complete!                   ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  URL:   http://localhost:3000                ║" -ForegroundColor Green
Write-Host "║  Login: admin / admin                       ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open your Streamlit app (streamlit run app.py)" -ForegroundColor White
Write-Host "  2. Upload a CSV dataset" -ForegroundColor White
Write-Host "  3. Go to 'Grafana Dashboard' page" -ForegroundColor White
Write-Host "  4. Click 'Sync Data to Grafana'" -ForegroundColor White
Write-Host ""
