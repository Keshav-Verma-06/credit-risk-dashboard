# 🚀 Complete Deployment Guide - Credit Risk Dashboard

## Option 1: Streamlit Cloud (RECOMMENDED - FREE)

### Step 1: Push Your Project to GitHub

```powershell
# 1. Initialize Git (if not done already)
git init

# 2. Add all files
git add .

# 3. Commit your code
git commit -m "Credit Risk Dashboard - Ready for deployment"

# 4. Create a new repository on GitHub
# Go to: https://github.com/new
# Repository name: credit-risk-dashboard
# Description: Enterprise Credit Risk Assessment Dashboard
# Visibility: Public (required for free Streamlit Cloud)
# Click "Create repository"

# 5. Connect and push to GitHub
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-dashboard.git
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. Click **"New app"** button
4. Fill in the form:
   - **Repository**: `YOUR_USERNAME/credit-risk-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy"**
6. Wait 2-3 minutes ⏳

### Your Dashboard Will Be Live At:
```
https://YOUR_USERNAME-credit-risk-dashboard.streamlit.app
```

### Step 3: Configure Data Source

Since we can't upload large files to GitHub, update the Google Drive link:

1. Make your dataset **publicly accessible** (Anyone with link can view)
2. Get the shareable link
3. In the dashboard sidebar, use "Load from URL"
4. Paste the direct download link:
   ```
   https://drive.google.com/uc?export=download&id=1QJYsKj4_MJCPE9nVBgpFPJAsevJ4Sd47
   ```

---

## Option 2: Hugging Face Spaces (FREE)

### Advantages:
- ✅ Free hosting
- ✅ More storage space
- ✅ ML/AI focused community

### Steps:

1. **Create account**: https://huggingface.co/join
2. **Create new Space**: https://huggingface.co/new-space
   - Name: `credit-risk-dashboard`
   - SDK: **Streamlit**
   - Visibility: Public
3. **Upload files**:
   - All `.py` files
   - `requirements.txt`
   - `README.md`
4. **Add data** (optional): Upload to Space or use URL

**Your app will be at**: `https://huggingface.co/spaces/YOUR_USERNAME/credit-risk-dashboard`

---

## Option 3: Render (FREE)

### Advantages:
- ✅ Free tier available
- ✅ More control over environment
- ✅ Supports custom domains

### Steps:

1. **Create account**: https://render.com/
2. **New Web Service** from GitHub repo
3. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Command**:
   ```bash
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
5. Click **"Create Web Service"**

**Note**: Free tier sleeps after 15 min of inactivity (wakes up when accessed)

---

## Option 4: Railway (FREE Tier)

### Steps:

1. **Sign up**: https://railway.app/
2. **New Project** → Deploy from GitHub
3. **Add start command** in railway.toml:
   ```toml
   [build]
   builder = "NIXPACKS"
   
   [deploy]
   startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
   ```

---

## Option 5: PythonAnywhere (FREE Basic)

### Limitations:
- Less ideal for Streamlit (better for Flask/Django)
- More configuration needed

### Alternative: Run as scheduled task or console

---

## 🔒 Handling Sensitive Data

### For Secrets (API keys, passwords):

#### On Streamlit Cloud:

1. Go to your app dashboard
2. Click **"Settings"** → **"Secrets"**
3. Add secrets in TOML format:
   ```toml
   [database]
   username = "your_username"
   password = "your_password"
   
   [api]
   key = "your_api_key"
   ```

#### In Code:
```python
import streamlit as st

# Access secrets
db_user = st.secrets["database"]["username"]
api_key = st.secrets["api"]["key"]
```

---

## 📦 Handling Large Files

### Your Model File (~50MB)

**Option A: Git LFS (Large File Storage)**

```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "Models/*.pkl"
git add .gitattributes
git commit -m "Track large files with Git LFS"
git push
```

**Option B: Cloud Storage**

Upload model to:
- Google Drive
- AWS S3
- Dropbox

Then download in code:
```python
import requests
import pickle

def load_model_from_url(url):
    response = requests.get(url)
    model = pickle.loads(response.content)
    return model
```

**Option C: Include Small Test Model**

Train a smaller model (fewer trees) just for demo purposes.

---

## 🎯 Pre-Deployment Checklist

Before deploying, verify:

- [ ] All dependencies in `requirements.txt`
- [ ] No absolute file paths (use relative paths)
- [ ] Data loading handles missing files gracefully
- [ ] No hardcoded credentials
- [ ] `.gitignore` excludes large files
- [ ] `README.md` has instructions
- [ ] App works locally: `streamlit run app.py`
- [ ] GitHub repository is public (for free hosting)

---

## 🐛 Common Deployment Issues

### Issue 1: "Module not found"
**Solution**: Add missing package to `requirements.txt`

### Issue 2: "File not found"
**Solution**: Use relative paths or load from URL

### Issue 3: "Out of memory"
**Solution**: 
- Reduce model size
- Use cloud storage for large files
- Optimize data loading

### Issue 4: "App keeps spinning"
**Solution**:
- Check logs in deployment platform
- Verify all dependencies install correctly
- Test locally first

---

## 📊 Monitoring Your Deployed App

### Streamlit Cloud:
- **Logs**: Available in app dashboard
- **Metrics**: View page visits
- **Errors**: Real-time error tracking

### Tips:
- Add error handling with try/except
- Use st.spinner() for loading states
- Add logging for debugging

---

## 🔄 Updating Your Deployed App

### Streamlit Cloud:
```powershell
# Make changes locally
git add .
git commit -m "Update: added new feature"
git push

# Streamlit Cloud automatically redeploys! ✨
```

### Manual Redeploy:
Click "Reboot app" in Streamlit Cloud dashboard

---

## 💰 Cost Comparison

| Platform | Free Tier | Limitations |
|----------|-----------|-------------|
| **Streamlit Cloud** | ✅ Yes | 1 private app, unlimited public |
| **Hugging Face** | ✅ Yes | Community tier |
| **Render** | ✅ Yes | Sleeps after 15 min |
| **Railway** | ✅ $5 credit | Then pay as you go |
| **Heroku** | ❌ No longer free | Min $7/month |

**Best Choice: Streamlit Cloud** for easiest deployment

---

## 🎓 Next Steps After Deployment

1. **Share your live URL** with stakeholders
2. **Gather feedback** and iterate
3. **Monitor usage** through platform analytics
4. **Add features** based on user needs
5. **Update documentation** with live URL

---

## 🆘 Need Help?

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum**: https://discuss.streamlit.io/
- **GitHub Issues**: For code-specific problems

---

## 🎉 Your Deployment Command Summary

```powershell
# Quick deployment to Streamlit Cloud:

# 1. Git setup
git init
git add .
git commit -m "Initial deployment"

# 2. Create repo on GitHub (https://github.com/new)

# 3. Push code
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-dashboard.git
git push -u origin main

# 4. Deploy on share.streamlit.io (2 minutes!)

# Done! Your app is live! 🚀
```

---

**Made with ❤️ by Keshav Verma | iitp_aiml_2506273**
