#!/usr/bin/env python3
"""
CourtCheck Diagnostic Tool
Identifies common issues preventing the tennis analysis model from running.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_python_version():
    """Check if Python version is 3.10+"""
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version_str} (✓ 3.10+ required)")
        return True
    else:
        print_error(f"Python {version_str} (✗ Need 3.10+)")
        return False

def check_required_files():
    """Check if all required project files exist"""
    print_header("Checking Required Files")
    
    required_files = {
        'Python Scripts': [
            'local_backend.py',
            'video_processor.py',
            'ball_detection.py',
            'court_detection_module.py',
            'stroke_classifier.py',
            'bounce_detection.py',
        ],
        'Model Weights': [
            'tracknet_weights.pt',
            'model_tennis_court_det.pt',
            'stroke_classifier_weights.pth',
            'bounce_detection_weights.cbm',
        ],
        'Configuration': [
            'requirements.txt',
            'coco_instances_results.json',
        ],
    }
    
    all_present = True
    missing_critical = []
    
    for category, files in required_files.items():
        print(f"\n{Colors.BOLD}{category}:{Colors.END}")
        for file in files:
            if os.path.exists(file):
                print_success(f"{file}")
            else:
                print_error(f"{file} - MISSING")
                all_present = False
                if category == 'Model Weights' or file in ['local_backend.py', 'video_processor.py']:
                    missing_critical.append(file)
    
    if missing_critical:
        print_warning(f"\n⚠️  Critical files missing: {', '.join(missing_critical)}")
        print_warning("These files are essential for the model to run!")
    
    return all_present

def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("Checking Python Dependencies")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV (opencv-python)'),
        ('detectron2', 'Detectron2'),
        ('catboost', 'CatBoost'),
        ('numpy', 'NumPy'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
    ]
    
    all_installed = True
    missing_packages = []
    
    for module_name, package_name in required_packages:
        try:
            if module_name == 'cv2':
                import cv2
            else:
                importlib.import_module(module_name)
            print_success(f"{package_name}")
        except ImportError:
            print_error(f"{package_name} - NOT INSTALLED")
            all_installed = False
            missing_packages.append(package_name)
    
    if missing_packages:
        print_warning(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print_warning("Install with: pip install -r requirements.txt")
    
    return all_installed

def check_file_permissions():
    """Check if Python files have proper permissions"""
    print_header("Checking File Permissions")
    
    python_files = ['local_backend.py', 'video_processor.py', 'run_local.py']
    all_ok = True
    
    for file in python_files:
        if os.path.exists(file):
            if os.access(file, os.R_OK):
                print_success(f"{file} is readable")
            else:
                print_error(f"{file} is NOT readable")
                all_ok = False
        else:
            print_warning(f"{file} does not exist")
    
    return all_ok

def analyze_common_errors():
    """Analyze Python files for common error patterns"""
    print_header("Analyzing Common Error Patterns")
    
    files_to_check = [
        'local_backend.py',
        'video_processor.py',
        'ball_detection.py',
        'court_detection_module.py',
    ]
    
    issues_found = []
    
    for file in files_to_check:
        if not os.path.exists(file):
            continue
            
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Check for common issues
            if 'import torch' in content and 'device' not in content:
                issues_found.append(f"{file}: Missing CUDA/CPU device configuration")
            
            if 'load_state_dict' in content or 'torch.load' in content:
                if 'map_location' not in content:
                    issues_found.append(f"{file}: torch.load() without map_location (may fail on CPU)")
            
            if '.pt' in content or '.pth' in content:
                # Check if file paths are hardcoded
                for i, line in enumerate(lines, 1):
                    if ('.pt"' in line or ".pt'" in line) and not os.path.exists(line.split('"')[1] if '"' in line else line.split("'")[1]):
                        issues_found.append(f"{file}:{i}: Hardcoded path to missing weight file")
    
    if issues_found:
        for issue in issues_found:
            print_warning(issue)
    else:
        print_success("No common error patterns detected")
    
    return len(issues_found) == 0

def check_port_availability():
    """Check if required ports are available"""
    print_header("Checking Port Availability")
    
    import socket
    
    ports = {
        8000: 'Backend (FastAPI)',
        3000: 'Frontend (React)',
    }
    
    all_available = True
    
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print_warning(f"Port {port} ({service}) - IN USE")
            all_available = False
        else:
            print_success(f"Port {port} ({service}) - Available")
    
    return all_available

def generate_report(results):
    """Generate final diagnostic report"""
    print_header("Diagnostic Report")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"Checks Passed: {passed_checks}/{total_checks}\n")
    
    if passed_checks == total_checks:
        print_success("✓ All checks passed! Your environment looks good.")
        print("\nYou can now try running:")
        print(f"{Colors.BOLD}  python local_backend.py{Colors.END}")
        print("or")
        print(f"{Colors.BOLD}  python run_local.py{Colors.END}")
    else:
        print_error("✗ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print(f"{Colors.BOLD}1. Install dependencies:{Colors.END}")
        print("   pip install -r requirements.txt")
        print(f"\n{Colors.BOLD}2. Download missing model weights{Colors.END}")
        print("   Check the project README for weight file links")
        print(f"\n{Colors.BOLD}3. Ensure Python 3.10+{Colors.END}")
        print("   python --version")

def main():
    """Run all diagnostic checks"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           CourtCheck Diagnostic Tool v1.0                 ║
    ║       Tennis Match Analysis - Troubleshooting             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    print(f"{Colors.END}")
    
    # Check if we're in the right directory
    if not os.path.exists('local_backend.py') and not os.path.exists('video_processor.py'):
        print_error("Error: Not in CourtCheck project directory!")
        print("Please navigate to your CourtCheck directory and run this script again.")
        sys.exit(1)
    
    # Run all checks
    results = {
        'Python Version': check_python_version(),
        'Required Files': check_required_files(),
        'Dependencies': check_dependencies(),
        'File Permissions': check_file_permissions(),
        'Common Errors': analyze_common_errors(),
        'Port Availability': check_port_availability(),
    }
    
    # Generate report
    generate_report(results)

if __name__ == '__main__':
    main()
