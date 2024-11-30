import PyInstaller.__main__
import os
import shutil

# Clean previous builds
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build'):
    shutil.rmtree('build')

# Create Python executable
PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--add-data', 'config.yaml:.',
    '--add-data', 'core;core',
    '--add-data', 'agents;agents',
    '--add-data', 'analysis;analysis',
    '--hidden-import', 'flask_cors',
    '--hidden-import', 'flask_socketio',
    '--hidden-import', 'engineio.async_drivers.threading',
    '--hidden-import', 'core.simulation',
    '--hidden-import', 'core.database',
    '--hidden-import', 'core.config',
    '--exclude-module', 'PyQt6',
    '--name', 'simulation_backend'
])

# Create python directory in electron build
os.makedirs('python', exist_ok=True)

# Copy executable and dependencies
shutil.copy(
    os.path.join('dist', 'simulation_backend.exe' if os.name == 'nt' else 'simulation_backend'),
    'python'
)
shutil.copy('config.yaml', 'python') 