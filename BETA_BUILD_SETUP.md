# Beta Build Setup Guide

This guide explains how to set up and use the GitHub Actions beta build workflow for Hearth.

## 🚀 Quick Start

### 1. Copy Workflow File to Main Repository

After merging this PR to `main`, copy the workflow file to the correct location:

```bash
# From repo root
cp desktop/installer/BUILD_BETA_WORKFLOW.yml .github/workflows/build-beta.yml
```

Or manually create `.github/workflows/build-beta.yml` with the contents from `desktop/installer/BUILD_BETA_WORKFLOW.yml`.

### 2. Trigger a Beta Build

#### Option A: Push a beta tag (Recommended)
```bash
git tag v0.2.7-beta.1
git push origin v0.2.7-beta.1
```

#### Option B: Manual trigger via GitHub UI
1. Go to **Actions** tab
2. Click **Build Beta Installers**
3. Click **Run workflow**
4. Enter version (e.g., `0.2.7-beta.1`)
5. Click **Run workflow**

## 📋 Workflow Details

### Triggers

The workflow automatically runs when:
- **Push beta tag**: `v*-beta*` or `beta-*` format
  - Examples: `v0.2.7-beta.1`, `beta-0.2.7`, `v1.0-beta`
- **Manual dispatch**: Click "Run workflow" in Actions tab

### Build Platforms

| Platform | Runner | Output | Architecture |
|----------|--------|--------|--------------|
| Windows | `windows-latest` | `.exe` (NSIS) | x86_64 |
| macOS | `macos-latest` | `.dmg` | ARM64 + x86_64 |
| Linux | `ubuntu-latest` | `.deb` + `.rpm` | x86_64 |

### Jobs Breakdown

#### 1. `build-windows`
- Installs NSIS
- Builds React UI
- Builds Rust backend
- Creates Windows NSIS installer
- Upload: `Hearth-VERSION-installer.exe`

#### 2. `build-macos`
- Builds for both ARM64 (Apple Silicon) and x86_64 (Intel)
- Creates universal DMG for each architecture
- Upload: `Hearth-VERSION-aarch64-apple-darwin.dmg` and `Hearth-VERSION-x86_64-apple-darwin.dmg`

#### 3. `build-linux`
- Builds DEB package (Debian/Ubuntu)
- Builds RPM package (Fedora/RHEL)
- Upload: `hearth_VERSION_amd64.deb` and `hearth-VERSION-1.x86_64.rpm`

#### 4. `create-beta-release`
- Downloads all artifacts from build jobs
- Creates GitHub Release (marked as prerelease)
- Attaches all installers to release
- Auto-generates release notes with installation instructions

## 📦 Generated Artifacts

After successful build, you'll have:

```
Release: Hearth 0.2.7-beta.1 (Beta)
├── Hearth-0.2.7-beta.1-installer.exe         (Windows)
├── Hearth-0.2.7-beta.1-aarch64-apple-darwin.dmg    (macOS ARM64)
├── Hearth-0.2.7-beta.1-x86_64-apple-darwin.dmg     (macOS Intel)
├── hearth_0.2.7-beta.1_amd64.deb             (Linux Debian)
└── hearth-0.2.7-beta.1-1.x86_64.rpm          (Linux RPM)
```

All artifacts are automatically uploaded to GitHub Releases and retained for 7 days in Actions.

## 🔧 Workflow Customization

### Change Artifact Retention

Edit `BUILD_BETA_WORKFLOW.yml`, find `retention-days` and change value:

```yaml
retention-days: 7  # Change to desired days
```

### Add Slack Notification

Add to `create-beta-release` job:

```yaml
- name: Notify Slack
  run: |
    curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
      -H 'Content-Type: application/json' \
      -d '{
        "text": "Hearth ${{ env.VERSION }} beta build complete! 🎉",
        "blocks": [{
          "type": "section",
          "text": {
            "type": "mrkdwn",
            "text": "*Hearth Beta Release*\nVersion: ${{ env.VERSION }}\nDownload: https://github.com/${{ github.repository }}/releases/tag/${{ env.VERSION }}"
          }
        }]
      }'
```

Then add secret: Settings → Secrets → `SLACK_WEBHOOK`

### Add Email Notification

Use `dawidd6/action-send-mail@v3`:

```yaml
- name: Send Email
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: ${{ secrets.EMAIL_SERVER }}
    server_port: ${{ secrets.EMAIL_PORT }}
    username: ${{ secrets.EMAIL_USER }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: Hearth ${{ env.VERSION }} Beta Released
    to: team@example.com
    from: releases@hearth.app
    body: |
      Hearth ${{ env.VERSION }} beta is now available!
      Download: https://github.com/${{ github.repository }}/releases/tag/${{ env.VERSION }}
```

## 🔐 Required Permissions

The workflow needs:
- ✅ `contents: read` (download code)
- ✅ `contents: write` (create releases)
- ✅ `actions: read` (access artifacts)

These are typically enabled by default. If builds fail with permission errors, go to:
- Settings → Actions → General
- Scroll to "Workflow permissions"
- Enable "Read and write permissions"

## 📊 Monitoring Builds

### View Build Progress
1. Go to **Actions** tab
2. Click **Build Beta Installers**
3. Click the running workflow
4. Watch real-time logs for each job

### Check Build Artifacts
1. Click on completed workflow
2. Scroll to "Artifacts"
3. Download individual platform installers

### View Release
1. Go to **Releases**
2. Find the beta release (marked with "Pre-release" badge)
3. Download installers or view release notes

## ⚡ Build Performance

Typical build times:
- **Windows**: 15-20 minutes
- **macOS**: 20-25 minutes (2 architectures)
- **Linux**: 10-15 minutes
- **Release creation**: 2-3 minutes

**Total**: ~45-60 minutes for all platforms

## 🐛 Troubleshooting

### Build Fails on Windows: NSIS Not Found
**Error**: `"C:\Program Files (x86)\NSIS\makensis.exe" not found`

**Solution**: NSIS installation may have failed. Check:
```yaml
- name: Install NSIS
  run: choco install nsis -y
```

Try using direct download:
```yaml
- name: Install NSIS
  run: |
    curl -L https://sourceforge.net/projects/nsis/files/NSIS%203/3.10/nsis-3.10-setup.exe/download -o nsis-setup.exe
    nsis-setup.exe /S
```

### Build Fails on macOS: Rust Target Missing
**Error**: `error: toolchain 'stable' does not support target 'aarch64-apple-darwin'`

**Solution**: Already handled in workflow, but verify:
```yaml
targets: aarch64-apple-darwin,x86_64-apple-darwin
```

### Build Fails on Linux: Missing Dependencies
**Error**: `libssl-dev not found` or similar

**Solution**: Dependencies are installed in workflow:
```yaml
sudo apt-get install -y build-essential libssl-dev libfontconfig1-dev rpm
```

If still fails, check Ubuntu version in runner:
```yaml
runs-on: ubuntu-latest  # Currently 22.04 LTS
```

### Release Creation Fails
**Error**: `GitHub API error: 422 Unprocessable Entity`

**Cause**: Release already exists for tag

**Solution**: Delete the tag and re-push:
```bash
git tag -d v0.2.7-beta.1
git push origin :v0.2.7-beta.1
git tag v0.2.7-beta.1
git push origin v0.2.7-beta.1
```

## 📝 Version Tagging Convention

Recommended beta versioning:

```
v0.2.7-beta.1      First beta
v0.2.7-beta.2      Second beta
v0.2.7-rc.1        Release candidate
v0.2.7              Final release
```

## 🔄 From Beta to Release

### 1. Test beta build thoroughly
- Download installers from release page
- Test on Windows, macOS, Linux
- Collect user feedback

### 2. Fix issues if needed
```bash
git tag v0.2.7-beta.2
git push origin v0.2.7-beta.2
```

### 3. Create final release
```bash
git tag v0.2.7
git push origin v0.2.7
```

The workflow will automatically create production release (not marked as prerelease).

## 📚 Useful Commands

### List all beta tags
```bash
git tag -l "*beta*"
```

### Delete a tag
```bash
git tag -d v0.2.7-beta.1
git push origin :v0.2.7-beta.1
```

### View workflow runs
```bash
gh run list --workflow=build-beta.yml
```

### Download artifacts locally
```bash
gh run download <run-id> -D ./artifacts
```

## 🎯 Next Steps

1. ✅ Create `.github/workflows/build-beta.yml` from `BUILD_BETA_WORKFLOW.yml`
2. ✅ Merge this PR to `main`
3. ✅ Push a beta tag: `git tag v0.2.7-beta.1 && git push origin v0.2.7-beta.1`
4. ✅ Watch the workflow run in Actions tab
5. ✅ Download and test installers from Releases
6. ✅ Share with beta testers!

## 💡 Tips

- **Keep it green**: Check workflow status before major releases
- **Test locally**: Always test installers before pushing tags
- **Document changes**: Update release notes with bug fixes
- **Communicate**: Notify team when beta is ready
- **Iterate fast**: Beta is for testing, don't overthink it

## 🆘 Need Help?

- Check workflow logs: Actions → [workflow] → [job] → view all steps
- GitHub Actions docs: https://docs.github.com/en/actions
- NSIS docs: https://nsis.sourceforge.io/
- Tauri docs: https://tauri.app/
