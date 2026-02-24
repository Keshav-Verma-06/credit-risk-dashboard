"""
🚀 Quick Deploy Script - Push to GitHub
This script helps you push your dashboard to GitHub for Streamlit Cloud deployment

Run: python deploy.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n📌 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def check_git():
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except:
        print("❌ Git is not installed!")
        print("Download from: https://git-scm.com/downloads")
        return False

def main():
    print("=" * 70)
    print("  🚀 GitHub Deployment Helper")
    print("=" * 70)
    
    # Check git
    if not check_git():
        return
    
    print("\n📋 This script will help you push your code to GitHub")
    print("   Then you can deploy on Streamlit Cloud (share.streamlit.io)")
    
    # Get GitHub username and repo name
    print("\n" + "=" * 70)
    print("Step 1: GitHub Repository Information")
    print("=" * 70)
    
    github_username = input("\n👤 Enter your GitHub username: ").strip()
    repo_name = input("📁 Enter repository name (default: credit-risk-dashboard): ").strip()
    
    if not repo_name:
        repo_name = "credit-risk-dashboard"
    
    if not github_username:
        print("❌ GitHub username is required!")
        return
    
    print(f"\n✅ Repository will be: https://github.com/{github_username}/{repo_name}")
    
    # Initialize git
    print("\n" + "=" * 70)
    print("Step 2: Initialize Git Repository")
    print("=" * 70)
    
    # Check if already initialized
    if os.path.exists('.git'):
        print("✅ Git repository already initialized")
    else:
        if not run_command('git init', "Initializing Git repository"):
            return
    
    # Create .gitignore if not exists (should already exist)
    if not os.path.exists('.gitignore'):
        print("⚠️  .gitignore not found, creating one...")
        with open('.gitignore', 'w') as f:
            f.write("__pycache__/\n*.pyc\n.env\n*.pkl\ndata/*.csv\n")
    
    # Add files
    print("\n" + "=" * 70)
    print("Step 3: Add Files to Git")
    print("=" * 70)
    
    if not run_command('git add .', "Adding all files"):
        return
    
    # Commit
    print("\n" + "=" * 70)
    print("Step 4: Commit Changes")
    print("=" * 70)
    
    commit_message = input("\n💬 Enter commit message (default: Initial commit): ").strip()
    if not commit_message:
        commit_message = "Initial commit - Credit Risk Dashboard"
    
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        print("⚠️  Nothing to commit or commit failed")
        # Continue anyway
    
    # Set branch to main
    print("\n" + "=" * 70)
    print("Step 5: Set Branch to Main")
    print("=" * 70)
    
    run_command('git branch -M main', "Setting branch to main")
    
    # Add remote
    print("\n" + "=" * 70)
    print("Step 6: Connect to GitHub")
    print("=" * 70)
    
    print(f"\n🔗 Before proceeding, create a repository on GitHub:")
    print(f"   1. Go to: https://github.com/new")
    print(f"   2. Repository name: {repo_name}")
    print(f"   3. Make it PUBLIC (required for free Streamlit Cloud)")
    print(f"   4. Do NOT initialize with README")
    print(f"   5. Click 'Create repository'")
    
    input("\n✅ Press Enter once you've created the repository on GitHub...")
    
    remote_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    # Check if remote already exists
    result = subprocess.run(['git', 'remote'], capture_output=True, text=True)
    
    if 'origin' in result.stdout:
        print("⚠️  Remote 'origin' already exists, removing and re-adding...")
        run_command('git remote remove origin', "Removing old remote")
    
    if not run_command(f'git remote add origin {remote_url}', "Adding GitHub remote"):
        return
    
    # Push to GitHub
    print("\n" + "=" * 70)
    print("Step 7: Push to GitHub")
    print("=" * 70)
    
    print("\n🚀 Pushing your code to GitHub...")
    print("   (You may be asked for GitHub credentials)")
    
    if not run_command('git push -u origin main', "Pushing to GitHub"):
        print("\n❌ Push failed. Common issues:")
        print("   1. Check your GitHub credentials")
        print("   2. Make sure repository exists on GitHub")
        print("   3. Try: git push -u origin main --force (if needed)")
        return
    
    # Success!
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Your code is on GitHub!")
    print("=" * 70)
    
    print(f"\n🔗 GitHub Repository: https://github.com/{github_username}/{repo_name}")
    print(f"\n📊 Next Steps - Deploy on Streamlit Cloud:")
    print(f"   1. Go to: https://share.streamlit.io")
    print(f"   2. Sign in with GitHub")
    print(f"   3. Click 'New app'")
    print(f"   4. Repository: {github_username}/{repo_name}")
    print(f"   5. Branch: main")
    print(f"   6. Main file: app.py")
    print(f"   7. Click 'Deploy'")
    print(f"\n🎉 Your dashboard will be live in 2-3 minutes at:")
    print(f"   https://{github_username}-{repo_name}.streamlit.app")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nYou can deploy manually following DEPLOYMENT_GUIDE.md")
