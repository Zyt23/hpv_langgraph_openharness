$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Force UTF-8 for PowerShell <-> native process piping to avoid Chinese mojibake.
chcp 65001 > $null
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
[System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
        }
    }
}

$orKey = [System.Environment]::GetEnvironmentVariable("OPENROUTER_API_KEY", "Process")
$oaKey = [System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
if (($orKey) -and (-not $oaKey)) {
    [System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $orKey, "Process")
}

[System.Environment]::SetEnvironmentVariable("OPENHARNESS_CONFIG_DIR", "$RootDir\.openharness", "Process")
[System.Environment]::SetEnvironmentVariable("OPENHARNESS_DATA_DIR", "$RootDir\.openharness\data", "Process")
[System.Environment]::SetEnvironmentVariable("OPENHARNESS_LOGS_DIR", "$RootDir\.openharness\logs", "Process")
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "$RootDir\src", "Process")
[System.Environment]::SetEnvironmentVariable("HTTP_PROXY", $null, "Process")
[System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", $null, "Process")
[System.Environment]::SetEnvironmentVariable("ALL_PROXY", $null, "Process")
[System.Environment]::SetEnvironmentVariable("http_proxy", $null, "Process")
[System.Environment]::SetEnvironmentVariable("https_proxy", $null, "Process")
[System.Environment]::SetEnvironmentVariable("all_proxy", $null, "Process")

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found in PATH"
}

if (-not (Get-Command openharness -ErrorAction SilentlyContinue)) {
    throw "OpenHarness CLI not found in PATH"
}

python -m hpv_agent.run_agent --config configs/hpv_openharness.yaml
