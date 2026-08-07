$ErrorActionPreference = "Stop"

$appDir = "c:\Users\gunas\OneDrive\Desktop\data-analytics-ml-app"
$grafanaPath = "$appDir\grafana_runtime\grafana-v11.1.0"

Write-Host "Updating Grafana provisioning configuration..."

# Datasources
$dsSource = "$appDir\grafana\provisioning\datasources"
$dsDest = "$grafanaPath\conf\provisioning\datasources"
if (-Not (Test-Path $dsDest)) { New-Item -ItemType Directory -Path $dsDest -Force | Out-Null }
Copy-Item -Path "$dsSource\*" -Destination $dsDest -Force
Write-Host "Datasource YAML updated."

# Dashboards YAML
$dbProvSource = "$appDir\grafana\provisioning\dashboards"
$dbProvDest = "$grafanaPath\conf\provisioning\dashboards"
if (-Not (Test-Path $dbProvDest)) { New-Item -ItemType Directory -Path $dbProvDest -Force | Out-Null }
Copy-Item -Path "$dbProvSource\*" -Destination $dbProvDest -Force
Write-Host "Dashboard YAML updated."

# Dashboards JSON
$dashSource = "$appDir\grafana\dashboards"
$dashDest = "$grafanaPath\public\dashboards\dataml"
if (-Not (Test-Path $dashDest)) { New-Item -ItemType Directory -Path $dashDest -Force | Out-Null }
Copy-Item -Path "$dashSource\*" -Destination $dashDest -Recurse -Force
Write-Host "Dashboard JSON files updated."

# Restart Grafana
Write-Host "Restarting Grafana server..."
Get-Process -Name "grafana-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Start-Process -FilePath "$grafanaPath\bin\grafana-server.exe" -WorkingDirectory "$grafanaPath" -WindowStyle Hidden
Write-Host "Grafana restarted successfully!"
