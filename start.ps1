# Windows equivalent of start.sh. Run in PowerShell:  .\start.ps1
# (If scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "x  $m" -ForegroundColor Red; exit 1 }

# 1. Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die "Chưa cài Docker Desktop: https://docs.docker.com/get-docker/" }
docker info *> $null; if ($LASTEXITCODE -ne 0) { Die "Docker chưa chạy. Mở Docker Desktop rồi chạy lại." }

# 2. .env
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Say "Đã tạo .env từ .env.example" }
$envVars = @{}
Get-Content .env | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$' } | ForEach-Object {
  $envVars[$Matches[1]] = $Matches[2].Trim()
}
$LLM   = if ($envVars.LLM_MODEL)   { $envVars.LLM_MODEL }   else { "qwen3:4b" }
$EMBED = if ($envVars.EMBED_MODEL) { $envVars.EMBED_MODEL } else { "bge-m3" }
$PORT  = if ($envVars.APP_PORT)    { $envVars.APP_PORT }    else { "8080" }
$default = "http://ollama:11434"

# 3. Hardware / where Ollama runs
$compose = @("compose", "-f", "docker-compose.yml")
$external = $envVars.OLLAMA_URL -and $envVars.OLLAMA_URL -ne $default
if ($external) {
  $env:OLLAMA_URL = $envVars.OLLAMA_URL
  Say "Dùng Ollama tại $($env:OLLAMA_URL) (từ .env)"
} else {
  $env:OLLAMA_URL = $default
  $compose += @("--profile", "ollama")
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $compose += @("-f", "docker-compose.gpu.yml")
    Say "Phát hiện GPU NVIDIA -> bật GPU cho Ollama"
  } else {
    Say "Không có GPU NVIDIA -> chạy CPU (chậm hơn nhưng vẫn dùng được)"
  }
}

# 4. Free port for the UI (another stack may already use 8080)
function Test-PortBusy($p) {
  $c = New-Object Net.Sockets.TcpClient
  try { return $c.ConnectAsync("127.0.0.1", [int]$p).Wait(300) } catch { return $false } finally { $c.Dispose() }
}
$appRunning = docker @compose ps -q app 2>$null
if (-not $appRunning -and (Test-PortBusy $PORT)) {
  $new = $null
  foreach ($p in ([int]$PORT + 1)..([int]$PORT + 50)) { if (-not (Test-PortBusy $p)) { $new = $p; break } }
  if (-not $new) { Die "Cổng $PORT đang bận và không tìm được cổng trống. Đặt APP_PORT trong .env." }
  $content = Get-Content .env
  if ($content -match '^APP_PORT=') { $content = $content -replace '^APP_PORT=.*', "APP_PORT=$new" } else { $content += "APP_PORT=$new" }
  Set-Content .env $content
  Warn "Cổng $PORT đang được chương trình khác dùng -> dùng cổng $new (đã lưu vào .env)"
  $PORT = "$new"
}
$env:APP_PORT = $PORT

# 5. Start services
Say "Khởi động services (lần đầu sẽ build image, mất vài phút)..."
docker @compose up -d --build --wait
if ($LASTEXITCODE -ne 0) { Die "docker compose up thất bại" }

# 6. Models
function Pull-Model($m) {
  if ($external) {
    $tags = Invoke-RestMethod "$($env:OLLAMA_URL)/api/tags"
    if ($tags.models.name -contains $m -or $tags.models.name -contains "$m`:latest") { Say "Model $m đã có"; return }
    Say "Tải model $m (chỉ lần đầu)..."
    Invoke-RestMethod -Method Post "$($env:OLLAMA_URL)/api/pull" -Body (@{model=$m; stream=$false} | ConvertTo-Json) | Out-Null
    return
  }
  $list = docker @compose exec -T ollama ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] }
  if ($list -contains $m -or $list -contains "$m`:latest") { Say "Model $m đã có"; return }
  Say "Tải model $m (chỉ lần đầu)..."
  docker @compose exec ollama ollama pull $m
  if ($LASTEXITCODE -ne 0) { Die "Không tải được model $m" }
}
Pull-Model $EMBED
Pull-Model $LLM

# 7. Ingest
Say "Ingest tài liệu trong data\ ..."
docker @compose exec -T app python -m app.ingest.run
if ($LASTEXITCODE -ne 0) { Die "Ingest thất bại" }

# 8. Ready
try {
  Invoke-RestMethod "http://localhost:$PORT/health" | Out-Null
  Write-Host "`n✅ Mở http://localhost:$PORT" -ForegroundColor Green
} catch {
  Warn "Hệ thống chưa sẵn sàng hoàn toàn, xem: http://localhost:$PORT/health"
}
Write-Host "   Dừng: .\stop.ps1"
Start-Process "http://localhost:$PORT"
