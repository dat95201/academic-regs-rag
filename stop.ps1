# Stop all services. Volumes (pulled models, Qdrant data) are kept.
Set-Location $PSScriptRoot
docker compose -f docker-compose.yml --profile ollama down
Write-Host "Đã dừng. Model và dữ liệu Qdrant vẫn được giữ trong Docker volume."
