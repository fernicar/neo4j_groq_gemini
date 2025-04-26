# main.py
import argparse
import os
import glob
import importlib.util
import sys
import time

def find_latest_core_module(directory="."):
    """Finds the latest coreYYYYMMDD.py file."""
    core_files = sorted(glob.glob(os.path.join(directory, "core*.py")))
    if not core_files:
        return None
    # Simple sort by name should work for YYYYMMDD format
    latest_core_path = core_files[-1]
    return latest_core_path

def find_latest_ini_file(directory="approach"):
    """Finds the latest .ini file in the specified directory."""
    if not os.path.exists(directory):
        return None
    ini_files = glob.glob(os.path.join(directory, "*.ini"))
    if not ini_files:
        return None
    # Sort by modification time
    latest_ini_path = max(ini_files, key=os.path.getmtime)
    return latest_ini_path

def main():
    parser = argparse.ArgumentParser(description="Run Knowledge Graph Consistency PoC tests.")
    parser.add_argument("--test", type=int, default=1,
                        help="Start running tests from this number.")
    parser.add_argument("--approach", type=str, default=None,
                        help="Specify the configuration .ini file (e.g., default.ini).")
    parser.add_argument("--use-file-response", action="store_true",
                        help="Use file-based response instead of making API calls.")
    parser.add_argument("--file-response-path", type=str, default="Raw Response Text.txt",
                        help="Path to the file containing the raw response text.")
    args = parser.parse_args()

    # 1. Find the latest core module
    latest_core_path = find_latest_core_module()
    if not latest_core_path:
        print("Error: No coreYYYYMMDD.py file found.")
        sys.exit(1)

    print(f"Using core logic from: {latest_core_path}")

    # 2. Determine the configuration file
    config_path = args.approach
    if config_path is None:
        latest_ini_path = find_latest_ini_file()
        if latest_ini_path:
            config_path = latest_ini_path
            print(f"No --approach specified. Using latest config: {config_path}")
        else:
            print("Error: No --approach specified and no .ini files found in ./approach/.")
            sys.exit(1)
    else:
        config_path = os.path.join("approach", config_path)
        if not os.path.exists(config_path):
            print(f"Error: Specified config file not found: {config_path}")
            sys.exit(1)
        print(f"Using specified config: {config_path}")

    # 3. Load the core module dynamically
    module_name = os.path.basename(latest_core_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, latest_core_path)
    if spec is None:
         print(f"Error: Could not load module spec for {latest_core_path}")
         sys.exit(1)
    core_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = core_module
    try:
        spec.loader.exec_module(core_module)
    except Exception as e:
        print(f"Error loading core module {latest_core_path}: {e}")
        sys.exit(1)


    # 4. Run the tests from the core module
    if hasattr(core_module, 'run_tests'):
        core_module.run_tests(
            start_test_number=args.test,
            config_path=config_path,
            use_file_response=args.use_file_response,
            file_response_path=args.file_response_path
        )
    else:
        print(f"Error: Core module {latest_core_path} does not have a 'run_tests' function.")
        sys.exit(1)

if __name__ == "__main__":
    main()