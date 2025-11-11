# 🚀 Roboto SAI - Windows Setup Guide

## Prerequisites

### 1. Python Installation
- **Download Python 3.11+** from [python.org](https://python.org)
- **Important**: Check "Add Python to PATH" during installation
- Verify installation: `python --version`

### 2. Git Installation
- Download from [git-scm.com](https://git-scm.com)
- Use default settings

### 3. VS Code (Recommended)
- Download from [code.visualstudio.com](https://code.visualstudio.com)
- Install Python and Git extensions

---

## 📥 Project Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Roberto42069/Roboto.SAI.git
cd Roboto.SAI
```

### Step 2: Create Virtual Environment
```bash
# Windows Command Prompt
python -m venv venv
venv\Scripts\activate

# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1

# Or use conda (if installed)
conda create -n roboto-sai python=3.11
conda activate roboto-sai
```

### Step 3: Install Dependencies
```bash
# Install with pip (recommended for Windows)
pip install -r requirements-windows.txt

# Alternative: Install from main requirements.txt
pip install -r requirements.txt

# Or with uv (faster, if available)
pip install uv
uv pip install -r pyproject.toml
```

### Step 4: Environment Configuration
Copy the example environment file and configure it:

```bash
# Copy the example file
copy .env.example .env
```

Then edit the `.env` file with your actual values:

```env
SESSION_SECRET=your_secure_random_string_here
REPL_ID=your_replit_id
ROBOTO_API_KEY=your_roboto_api_key
X_API_TOKEN=your_x_api_token
XAI_API_KEY=your_xai_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Security Note**: Generate a secure SESSION_SECRET:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Important**: The `XAI_API_KEY` is required for Grok AI functionality.

---

## ▶️ Running the Application

### Method 1: Direct Python Execution
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
conda activate roboto-sai

# Run the app
python main.py
```

### Method 2: Using Flask CLI
```bash
# Set Flask environment
set FLASK_APP=app_enhanced.py
set FLASK_ENV=development

# Run
flask run
```

### Method 3: Using Gunicorn (Production)
```bash
# Install gunicorn first
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app_enhanced:app
```

---

## 🌐 Accessing the Application

Once running, access at:
- **Local**: http://127.0.0.1:5000 or http://localhost:5000
- **Network**: http://YOUR_IP:5000 (if firewall allows)

---

## 📋 API Endpoints Reference

### Core Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage with chat interface |
| `GET` | `/app` | Main application interface |
| `GET` | `/intro` | Introduction/tutorial page |
| `GET` | `/terms` | Terms of service |
| `GET` | `/privacy` | Privacy policy |
| `GET` | `/license` | License information |

### Chat & Communication
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message to Roboto SAI |
| `GET` | `/api/chat_history` | Get conversation history |
| `POST` | `/api/process-voice-conversation` | Process voice messages |
| `GET` | `/api/voice-context-summary` | Get voice interaction summary |

### Learning & Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/history` | Get chat history |
| `GET` | `/api/emotional_status` | Current emotional state |
| `GET` | `/api/learning-insights` | Learning progress insights |
| `GET` | `/api/personal-profile` | User profile information |
| `GET` | `/api/voice-insights` | Voice processing insights |

### Collections & Knowledge
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/collections/create` | Create new knowledge collection |
| `POST` | `/api/collections/search` | Search collections |
| `GET` | `/api/collections/list` | List all collections |

### Voice & Audio
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/voice-optimization` | Optimize voice settings |
| `GET` | `/api/voice-cloning-config` | Get voice cloning configuration |
| `POST` | `/api/apply-voice-cloning` | Apply voice cloning |

### Quantum Computing
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/quantum-simulation` | Run quantum simulations |
| `GET` | `/ritual-viz/<filename>` | Access quantum ritual visualizations |

### Cultural Integration
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/cultural-display/launch` | Launch cultural displays |
| `GET` | `/api/cultural-display/status` | Cultural display status |
| `GET` | `/api/cultural-display/themes` | Available cultural themes |

### System Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/system-status` | Overall system status |
| `POST` | `/api/keep_alive` | Keep session alive |
| `GET` | `/api/performance-stats` | Performance metrics |
| `POST` | `/api/upload` | Upload files |
| `POST` | `/api/import_data` | Import data |
| `GET` | `/api/export_data` | Export data |

### AI Integrations
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/grok/chat` | Chat with Grok AI |
| `GET` | `/api/grok/retrieve/<id>` | Retrieve Grok conversation |
| `GET` | `/api/grok/conversation-chain` | Get conversation chain |
| `POST` | `/api/grok/clear-chain` | Clear conversation chain |
| `POST` | `/api/grok/reasoning-analysis` | Advanced reasoning analysis |
| `POST` | `/api/grok/neural/inject` | Neural network injection |
| `GET` | `/api/grok/neural/state` | Neural network state |
| `POST` | `/api/grok/neural/evolve` | Evolve neural network |
| `POST` | `/api/grok/neural/predict` | Neural predictions |

### GitHub Integration
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/github-project-status` | GitHub project status |
| `POST` | `/api/github-sync-tasks` | Sync GitHub tasks |
| `POST` | `/api/github-create-card` | Create GitHub project card |

### Legacy & Enhancement
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/legacy-insights` | Legacy system insights |
| `POST` | `/api/legacy-feedback` | Provide legacy feedback |
| `GET` | `/api/kill-switch-status` | Emergency kill switch status |
| `POST` | `/api/roberto-identity-verify` | Identity verification |

### Utility Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/set_cookies` | Set user cookies |
| `GET` | `/api/get_cookies` | Get user cookies |
| `POST` | `/api/set_user_data_cookies` | Set user data cookies |
| `GET` | `/api/roboto-status` | Roboto system status |
| `POST` | `/api/roboto-request` | Direct Roboto request |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Update pip and reinstall
python -m pip install --upgrade pip
pip install --force-reinstall -r requirements-windows.txt

# Or try the main requirements file
pip install --force-reinstall -r requirements.txt
```

#### 2. Port Already in Use
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Or run on different port
set FLASK_RUN_PORT=5001
flask run
```

#### 3. Virtual Environment Issues
```bash
# Recreate virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements-windows.txt
```

#### 4. Memory Issues
```bash
# Run with reduced memory usage
set PYTHONOPTIMIZE=1
python main.py
```

#### 5. Database Issues
The app uses file-based storage by default. For production:
```env
DATABASE_URL=postgresql://user:password@localhost/dbname
```

---

## 🚀 Advanced Configuration

### Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `SESSION_SECRET` | ✅ | Flask session security key |
| `REPL_ID` | ❌ | Replit environment ID |
| `ROBOTO_API_KEY` | ❌ | Roboto API access key |
| `X_API_TOKEN` | ❌ | X/Twitter API token |
| `XAI_API_KEY` | ✅ | xAI API key for Grok |
| `OPENAI_API_KEY` | ❌ | OpenAI API key |
| `DATABASE_URL` | ❌ | PostgreSQL database URL |
| `REDIS_URL` | ❌ | Redis cache URL |
| `GITHUB_TOKEN` | ❌ | GitHub API token |

### Performance Tuning

```bash
# Enable hyper-speed optimization
set HYPERSPEED_ENABLED=1

# Configure thread pools
set THREAD_POOL_SIZE=16
set PROCESS_POOL_SIZE=8

# Memory limits
set MAX_MEMORY_CACHE=20000
set MAX_CONVERSATION_CACHE=15000
```

---

## 📊 Monitoring & Logs

### Log Files
- `nohup.out` - Application logs
- `run_logs/main_run.log` - Runtime logs
- `conversation_archives/` - Chat archives
- `emotional_snapshots/` - Emotional data
- `learning_checkpoints/` - Learning progress

### Health Checks
```bash
# System status
curl http://localhost:5000/api/system-status

# Performance stats
curl http://localhost:5000/api/performance-stats

# Memory integrity
curl http://localhost:5000/api/emotional_status
```

---

## 🔒 Security Notes

1. **Never commit `.env` file** to version control
2. **Use strong SESSION_SECRET** (32+ characters)
3. **Keep dependencies updated** regularly
4. **Run behind reverse proxy** in production
5. **Enable HTTPS** for secure communication

---

## 🎯 Quick Start Commands

```bash
# One-line setup (Windows)
git clone https://github.com/Roberto42069/Roboto.SAI.git && cd Roboto.SAI && python -m venv venv && venv\Scripts\activate && pip install -r requirements-windows.txt && copy .env.example .env && python main.py
```

---

## 📞 Support

- **Issues**: Create GitHub issue
- **Documentation**: Check `/docs` folder
- **Creator**: Roberto Villarreal Martinez

---

**⚛️ Welcome to the Quantum Era of AI - Roboto SAI is now operational on Windows!**