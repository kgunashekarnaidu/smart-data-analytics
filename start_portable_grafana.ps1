# Portable Grafana Setup Script (No Admin Privileges Required)
$ErrorActionPreference = "Stop"

$appDir = "c:\Users\gunas\OneDrive\Desktop\data-analytics-ml-app"
$zipUrl = "https://dl.grafana.com/oss/release/grafana-11.1.0.windows-amd64.zip"
$zipPath = "$env:TEMP\grafana-11.1.0.zip"
$destDir = "$appDir\grafana_runtime"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DataML Pro - Portable Grafana Setup   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path "$destDir\grafana-v11.1.0\bin\grafana-server.exe")) {
    if (-not (Test-Path $zipPath)) {
        Write-Host "[1/4] Downloading Grafana 11.1.0 Portable Zip..." -ForegroundColor Yellow
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Write-Host "[OK] Downloaded." -ForegroundColor Green
    }

    Write-Host "[2/4] Extracting Grafana Zip..." -ForegroundColor Yellow
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    Expand-Archive -Path $zipPath -DestinationPath $destDir -Force
    Write-Host "[OK] Extracted to $destDir." -ForegroundColor Green
} else {
    Write-Host "[OK] Portable Grafana already extracted." -ForegroundColor Green
}

$grafanaPath = "$destDir\grafana-v11.1.0"

# --- Configure custom.ini ---
Write-Host "[3/4] Configuring Grafana settings for iframe embedding..." -ForegroundColor Yellow
$customIni = "$grafanaPath\conf\custom.ini"
$customIniContent = @"
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

# --- Copy Provisioning Files ---
Write-Host "[4/4] Provisioning datasources and dashboards..." -ForegroundColor Yellow
$dsSource = "$appDir\grafana\provisioning\datasources"
$dsDest = "$grafanaPath\conf\provisioning\datasources"
if (-Not (Test-Path $dsDest)) { New-Item -ItemType Directory -Path $dsDest -Force | Out-Null }
Copy-Item -Path "$dsSource\*" -Destination $dsDest -Force

$dbProvSource = "$appDir\grafana\provisioning\dashboards"
$dbProvDest = "$grafanaPath\conf\provisioning\dashboards"
if (-Not (Test-Path $dbProvDest)) { New-Item -ItemType Directory -Path $dbProvDest -Force | Out-Null }
Copy-Item -Path "$dbProvSource\*" -Destination $dbProvDest -Force

$dashSource = "$appDir\grafana\dashboards"
$dashDest = "$grafanaPath\public\dashboards\dataml"
if (-Not (Test-Path $dashDest)) { New-Item -ItemType Directory -Path $dashDest -Force | Out-Null }
Copy-Item -Path "$dashSource\*" -Destination $dashDest -Recurse -Force

Write-Host "[OK] Provisioning complete." -ForegroundColor Green

# --- Start Grafana Server process ---
Write-Host "Starting Grafana server on port 3000..." -ForegroundColor Yellow
Start-Process -FilePath "$grafanaPath\bin\grafana-server.exe" -WorkingDirectory "$grafanaPath" -WindowStyle Hidden
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Portable Grafana Started on Port 3000! " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
