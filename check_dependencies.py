"""
Dependency Checker for CryptoSense
Verifies all required packages are installed correctly.
"""

import sys

print("="*70)
print("  CryptoSense - Dependency Checker")
print("="*70 + "\n")

required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'yfinance': 'yfinance',
    'requests': 'requests',
    'sklearn': 'scikit-learn',
    'tensorflow': 'tensorflow',
    'plotly': 'plotly',
    'streamlit': 'streamlit',
    'scipy': 'scipy'
}

all_installed = True

for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        print(f"✅ {package_name:20s} - Installed")
    except ImportError:
        print(f"❌ {package_name:20s} - NOT FOUND")
        all_installed = False

print("\n" + "="*70)

if all_installed:
    print("✅ All dependencies are installed!")
    print("\nYou're ready to run CryptoSense!")
    print("\nNext steps:")
    print("  1. python run_pipeline.py        # Run complete pipeline")
    print("  2. streamlit run dashboard.py    # Launch dashboard")
else:
    print("❌ Some dependencies are missing!")
    print("\nTo install all dependencies, run:")
    print("  pip install -r requirements.txt")

print("="*70 + "\n")

# Check Python version
print(f"Python version: {sys.version}")
if sys.version_info >= (3, 8):
    print("✅ Python version is compatible (3.8+)")
else:
    print("⚠️  Python 3.8+ recommended (you have {}.{})".format(
        sys.version_info.major, sys.version_info.minor))

sys.exit(0 if all_installed else 1)
