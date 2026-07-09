# Deploy DataML Pro — GitHub + Streamlit Cloud

Follow these steps to upload the project to GitHub and deploy it online for free.

---

## Part 1 — Prepare the project (one time)

Open **Terminal** in Cursor (`Ctrl + `` ` ``) and run:

```powershell
cd C:\Users\gunas\data-analytics-ml-app
```

Initialize Git **inside the project folder** (not your home folder):

```powershell
git init
git branch -M main
```

Check what will be uploaded:

```powershell
git status
```

You should see only project files (`app.py`, `core/`, `requirements.txt`, etc.).  
You should **NOT** see your entire `Downloads`, `Documents`, or CSV files from other folders.

---

## Part 2 — Create a GitHub repository

1. Go to [https://github.com/new](https://github.com/new)
2. **Repository name:** `data-analytics-ml-app` (or any name you prefer)
3. **Description:** `Streamlit data analytics & ML dashboard for any CSV`
4. Choose **Public** (required for free Streamlit Cloud)
5. **Do NOT** check "Add a README" (you already have one)
6. Click **Create repository**

Copy the repository URL — it looks like:

```
https://github.com/YOUR_USERNAME/data-analytics-ml-app.git
```

---

## Part 3 — Upload files to GitHub

In the same terminal (project folder):

```powershell
cd C:\Users\gunas\data-analytics-ml-app

git add .
git commit -m "Initial commit: DataML Pro Streamlit app"

git remote add origin https://github.com/YOUR_USERNAME/data-analytics-ml-app.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### If GitHub asks you to sign in

- **Username:** your GitHub username  
- **Password:** use a **Personal Access Token** (not your GitHub password)

Create a token: GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token** → enable `repo` scope.

### If `git remote add` fails (remote already exists)

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/data-analytics-ml-app.git
git push -u origin main
```

---

## Part 4 — Deploy on Streamlit Community Cloud (free)

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub** account
3. Click **New app**
4. Fill in:

   | Field | Value |
   |-------|--------|
   | Repository | `YOUR_USERNAME/data-analytics-ml-app` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL (optional) | `dataml-pro` or leave default |

5. Click **Deploy**

Streamlit will install packages from `requirements.txt` and start your app.  
First deploy takes **2–5 minutes**.

Your live URL will look like:

```
https://YOUR_APP_NAME.streamlit.app
```

---

## Part 5 — Update the app after changes

Whenever you edit code locally:

```powershell
cd C:\Users\gunas\data-analytics-ml-app
git add .
git commit -m "Describe your change"
git push
```

Streamlit Cloud **automatically redeploys** when you push to `main`.

---

## What gets uploaded vs ignored

| Uploaded to GitHub | Not uploaded (local only) |
|--------------------|---------------------------|
| `app.py`, `core/`, `requirements.txt` | `artifacts/` (trained models) |
| `README.md`, `.streamlit/config.toml` | `logs/` (log files) |
| `.gitignore` | `.venv/` virtual environment |
| Empty folder placeholders | Large CSV datasets you upload in the app |

The app works on Streamlit Cloud because users **upload their own CSV** in the browser — no dataset files need to be in GitHub.

---

## Troubleshooting

### Push rejected / authentication failed
Use a Personal Access Token instead of password (see Part 3).

### Streamlit deploy fails on `requirements.txt`
Make sure `requirements.txt` is in the repo root and contains all packages.

### App crashes on Streamlit Cloud
Open the app → **Manage app** → **Logs** to see the error.

### Wrong files were staged (entire home folder)
You ran git from the wrong folder. Run these from the project folder only:

```powershell
cd C:\Users\gunas\data-analytics-ml-app
git init
git add .
git status
```

Only project files should appear before committing.

### `ModuleNotFoundError: No module named 'core'`
Ensure `app.py` is at the repo root and you deploy with main file path `app.py`.

---

## Optional — Deploy with Render or Railway

Streamlit Cloud is the easiest option. Alternatives:

- **Render:** create a Web Service, build `pip install -r requirements.txt`, start `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Railway / Hugging Face Spaces:** similar setup with Streamlit start command

For beginners, **Streamlit Community Cloud** is recommended.

---

## Quick checklist

- [ ] Git initialized in `data-analytics-ml-app` folder
- [ ] GitHub repo created (public)
- [ ] `git push` succeeded
- [ ] Streamlit Cloud app connected to repo
- [ ] Main file set to `app.py`
- [ ] Live URL opens and app loads
