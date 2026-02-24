"""
🚀 Credit Risk Dashboard - Startup Script
Run this script to set up and launch your dashboard quickly

Usage: python start.py
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 
        'xgboost', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    return missing

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ All dependencies installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def train_model():
    """Train a quick model if none exists"""
    models_folder = 'Models'
    
    if os.path.exists(models_folder):
        pkl_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
        if pkl_files:
            print(f"\n✅ Found existing model(s): {', '.join(pkl_files)}")
            return True
    
    print("\n⚠️  No trained model found!")
    response = input("Would you like to train a quick test model now? (y/n): ").lower()
    
    if response == 'y':
        print("\n🤖 Training model... (this may take 30-60 seconds)")
        try:
            subprocess.check_call([sys.executable, 'train_quick_model.py'])
            print("\n✅ Model trained successfully!")
            return True
        except Exception as e:
            print(f"\n❌ Error training model: {e}")
            return False
    else:
        print("\n⚠️  You'll need to upload a model manually in the dashboard")
        return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching Credit Risk Dashboard...")
    print("\n" + "="*70)
    print("  The dashboard will open in your browser shortly...")
    print("  Press Ctrl+C in this terminal to stop the server")
    print("="*70 + "\n")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. See you next time!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\n💡 Try running manually: streamlit run app.py")

def main():
    """Main startup flow"""
    print_header("💳 Credit Risk Assessment Dashboard - Setup & Launch")
    
    print("👋 Welcome! This script will help you set up and launch the dashboard.\n")
    
    # Step 1: Check dependencies
    print_header("Step 1: Dependency Check")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages now? (y/n): ").lower()
        
        if response == 'y':
            if not install_dependencies():
                print("\n❌ Setup failed. Please install dependencies manually:")
                print("   pip install -r requirements.txt")
                return
        else:
            print("\n⚠️  Cannot proceed without required packages.")
            print("Please install manually: pip install -r requirements.txt")
            return
    else:
        print("\n✅ All dependencies are installed!")
    
    # Step 2: Check for trained model
    print_header("Step 2: Model Check")
    train_model()
    
    # Step 3: Launch dashboard
    print_header("Step 3: Launch Dashboard")
    
    response = input("Ready to launch the dashboard? (y/n): ").lower()
    
    if response == 'y':
        launch_dashboard()
    else:
        print("\n👋 Setup complete! Run 'streamlit run app.py' when ready.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Run this script again when ready!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please try running manually: streamlit run app.py")
