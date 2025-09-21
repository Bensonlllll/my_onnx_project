import importlib

def check_version(package_name):
    try:
        pkg = importlib.import_module(package_name)
        version = getattr(pkg, '__version__', '找不到版本資訊')
        print(f"{package_name}: {version}")
    except ImportError:
        print(f"{package_name} 尚未安裝")

if __name__ == "__main__":
    packages = [
        "onnx",
        "onnxruntime",
        "tf2onnx",
        "tensorflow",
        "torch",
        "torchvision",
        "numpy"
    ]
    print("檢查套件版本：")
    for pkg in packages:
        check_version(pkg)
