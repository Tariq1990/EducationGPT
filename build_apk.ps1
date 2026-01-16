# Build APK Helper Script
Write-Host "Checking Flutter installation..."

$flutterExists = Get-Command flutter -ErrorAction SilentlyContinue

if (-not $flutterExists) {
    Write-Host "Error: Flutter is not installed or not in your PATH."
    Write-Host "Please install Flutter from: https://docs.flutter.dev/get-started/install/windows"
    exit 1
}

if (-not (Test-Path "mobile_app")) {
    Write-Host "Error: 'mobile_app' folder not found."
    exit 1
}

Set-Location "mobile_app"
Write-Host "Fetching dependencies..."
& flutter pub get

Write-Host "Building APK (Release mode)..."
& flutter build apk --release

if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Your APK is located at:"
    Write-Host "C:\Users\Engr.Tariq Jamal\Downloads\EMA_ML_model\mobile_app\build\app\outputs\flutter-apk\app-release.apk"
}
else {
    Write-Host "Build failed. Please check the errors above."
}
