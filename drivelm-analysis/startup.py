
import os
import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_main_py():
    main_py = Path("main.py")
    if not main_py.exists():
        print("❌ main.py not found in current directory")
        print("   Please ensure you're running this from the project root")
        return False
    print("✅ main.py found")
    return True

def check_data_paths():
    nuscenes_path = Path("../data/nusccens")
    drivelm_path = Path("../data/drivelm_data/train_sample.json")
    
    paths_status = {
        "NuScenes data": nuscenes_path.exists(),
        "DriveLM data": drivelm_path.exists()
    }
    
    all_exist = True
    for name, exists in paths_status.items():
        if exists:
            print(f"✅ {name} found")
        else:
            print(f"⚠️ {name} not found at expected location")
            all_exist = False
    
    if not all_exist:
        print("\n💡 Data paths can be configured in the Streamlit interface")
        print("   The app will still work, but you'll need to run the pipeline manually")
    
    return paths_status

def check_required_packages():
    required_packages = {
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'pathlib': None,  # Built-in
        'json': None,     # Built-in
        'logging': None   # Built-in
    }
    
    missing_packages = []
    
    print("📦 Checking required packages...")
    
    for package, install_name in required_packages.items():
        if install_name is None:  
            print(f"✅ {package} (built-in)")
            continue
            
        try:
            if package == 'PIL':
                import PIL
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(install_name or package)
    
    optional_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu'
    }
    
    print("\n🔧 Checking optional RAG packages...")
    rag_available = True
    
    for package, install_name in optional_packages.items():
        try:
            if package == 'faiss':
                import faiss
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️ {package} (optional for RAG)")
            rag_available = False
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False, rag_available
    
    if not rag_available:
        print("\n⚠️ Some RAG packages are missing - RAG functionality will be limited")
        print("💡 For full RAG support, install:")
        print("   pip install torch transformers sentence-transformers faiss-cpu")
    
    return True, rag_available

def check_project_structure():
    expected_dirs = ['data_analysis', 'RAG']
    expected_files = ['main.py', 'app.py']
    
    print("\n📁 Checking project structure...")
    
    all_good = True
    
    for directory in expected_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/ directory")
        else:
            print(f"⚠️ {directory}/ directory missing")
            if directory == 'data_analysis':
                all_good = False
    
    for file in expected_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    return all_good

def create_output_directory():
    output_dir = Path("results")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created output directory: {output_dir}")
    else:
        print(f"✅ Output directory exists: {output_dir}")

def provide_usage_instructions():
    print("\n" + "="*60)
    print("🚀 DRIVELM ANALYSIS SUITE - USAGE INSTRUCTIONS")
    print("="*60)
    print("\n1. 📊 PIPELINE TAB:")
    print("   • Click 'Run Main Pipeline' to execute the complete analysis")
    print("   • Monitor real-time progress and output")
    print("   • Wait for completion before using other features")
    
    print("\n2. 📈 DATA ANALYSIS & DASHBOARDS:")
    print("   • View generated findings and statistics")
    print("   • Explore interactive visualizations")
    print("   • Available after pipeline completion")
    
    print("\n3. 🤖 RAG SYSTEM:")
    print("   • Click 'Load RAG System' after pipeline completion")
    print("   • Ask questions about driving scenarios")
    print("   • Use 'Run Fine-tuning' for bonus points")
    
    print("\n4. 📊 EVALUATION:")
    print("   • View RAG system performance metrics")
    print("   • Compare before/after fine-tuning results")
    
    print("\n💡 TIPS:")
    print("   • Run pipeline first, then explore other tabs")
    print("   • RAG system requires pipeline completion")
    print("   • Fine-tuning is optional but earns bonus points")
    print("   • Use sample questions for quick testing")
    print("="*60)

def launch_streamlit():
    print("\n🚀 Launching Streamlit application...")
    print("   📱 Access at: http://localhost:8501")
    print("   🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        env = os.environ.copy()
        env.update({
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_PORT': '8501',
            'STREAMLIT_SERVER_ADDRESS': 'localhost',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        subprocess.run([
            sys.executable,'-m', 'streamlit', 'run', 'app.py',
            '--server.headless', 'true',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install with:")
        print("   pip install streamlit")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")

def main():
    """Main startup function."""
    print("🚗 DriveLM Analysis Suite - Startup Check")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 5
    
    if check_python_version():
        checks_passed += 1
    
    if check_main_py():
        checks_passed += 1
    
    data_paths = check_data_paths()
    if any(data_paths.values()):
        checks_passed += 1
        
    packages_ok, rag_available = check_required_packages()
    if packages_ok:
        checks_passed += 1
    
    if check_project_structure():
        checks_passed += 1
    
    create_output_directory()
    
    print(f"\n📊 STARTUP CHECK SUMMARY: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 4: 
        print("✅ System ready to launch!")
        
        if not rag_available:
            print("⚠️ RAG functionality will be limited due to missing packages")
        
        provide_usage_instructions()
        
        print("\n" + "="*60)
        launch_streamlit()        
    else:
        print("❌ Too many issues found. Please fix the problems above before launching.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install streamlit plotly pandas numpy Pillow")
        print("   • Ensure you're in the correct project directory")
        print("   • Check that main.py and app.py exist")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
    