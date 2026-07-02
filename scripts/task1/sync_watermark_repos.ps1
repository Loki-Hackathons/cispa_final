# Sync Task 1 watermark detector submodules to commits pinned in watermark_config.yaml.
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

git config core.longpaths true

git submodule sync --recursive
git submodule update --init --recursive `
  task_1_text_watermark/vendor/textseal `
  task_1_text_watermark/vendor/lm-watermarking `
  task_1_text_watermark/vendor/unigram-watermark

$Pins = @{
  "task_1_text_watermark/vendor/textseal" = "788fe8bff5cf086f0881928ce9a81aa08c21dff1"
  "task_1_text_watermark/vendor/lm-watermarking" = "82922516930c02f8aa322765defdb5863d07a00e"
  "task_1_text_watermark/vendor/unigram-watermark" = "b96cdb4d52771e3cbd543a9d9aeeaec8d0790ca2"
}

foreach ($entry in $Pins.GetEnumerator()) {
  $path = $entry.Key
  $commit = $entry.Value
  Write-Host "==> $path @ $($commit.Substring(0, 7))"
  git -C $path config core.longpaths true
  git -C $path fetch --quiet origin 2>$null
  git -C $path checkout -f $commit
  $actual = git -C $path rev-parse HEAD
  if ($actual -ne $commit) {
    throw "$path is at $actual, expected $commit"
  }
}

Write-Host ""
Write-Host "Submodules pinned:"
git submodule status task_1_text_watermark/vendor/textseal `
  task_1_text_watermark/vendor/lm-watermarking `
  task_1_text_watermark/vendor/unigram-watermark
