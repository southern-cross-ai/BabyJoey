import os
import importlib
import inspect

def import_classes_from_subfolders(base_package, base_path):
    """
    Recursively import all classes from all subfolders and their corresponding Python files.
    """
    for root, dirs, files in os.walk(base_path):
        # Get the relative module path by replacing '/' with '.' and removing base path
        relative_module = root.replace(base_path, "").replace(os.sep, ".").lstrip(".")
        module_prefix = f"{base_package}.{relative_module}" if relative_module else base_package
        
        for filename in files:
            if filename.endswith(".py") and filename != "__init__.py":
                # Remove the `.py` extension
                module_name = filename[:-3]
                
                # Construct the full module name
                full_module_name = f"{module_prefix}.{module_name}" if relative_module else f"{base_package}.{module_name}"
                
                # Dynamically import the module
                module = importlib.import_module(full_module_name)
                
                # Inspect the module and add classes to the global scope
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # Add class to globals so it can be accessed directly
                        globals()[name] = obj

# Call the function to import classes from all subfolders under 'src'
import_classes_from_subfolders("src", os.path.dirname(__file__))