# How to Generate your APK

Since your computer doesn't have the Flutter SDK installed yet, you have two simple ways to get your APK file.

### 🚀 Option 1: Fast Web Build (No Software Needed)
This is the easiest way to get an APK in 5 minutes:
1. Go to **[FlutLab.io](https://flutlab.io/)**.
2. Create a free account.
3. Upload the `mobile_app` folder located at:
   `c:\Users\Engr.Tariq Jamal\Downloads\EMA_ML_model\mobile_app`
4. Click **"Build APK"** (the hammer icon).
5. Download and install the `.apk` on your phone!

---

### 💻 Option 2: Local Build (Recommended for Developers)
If you want to build it on your computer, follow these steps:

#### 1. Install Flutter
If you haven't already, download the [Flutter SDK](https://docs.flutter.dev/get-started/install/windows).

#### 2. Run the Build Script
I've created a helper script for you. Open a terminal and run:
```powershell
./build_apk.ps1
```

---

### ⚠️ Important: URL Configuration
Before building, make sure to update the URL in your app code:
- **File**: `mobile_app/lib/main.dart`
- **Line**: 54
- **Action**: Replace `https://your-streamlit-app-url.streamlit.app` with your actual hosted URL.

> [!NOTE]
> For the APK to work outside your home, you must deploy the dashboard to a cloud service (like Streamlit Cloud) first.
