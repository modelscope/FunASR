import shutil
import subprocess


def install_requirements(requirements_path):
    try:
        result = pip_install_r(requirements_path)
        # check status
        if result.returncode == 0:
            print("install model requirements successfully")
            return True
        else:
            print("fail to install model requirements! ")
            print("error", result.stderr)
            return False
    except Exception as e:
        result = pip_install_r(requirements_path)
        # check status
        if result.returncode == 0:
            print("install model requirements successfully")
            return True
        else:
            print("fail to install model requirements! ")
            print("error", result.stderr)
            return False


def pip_install_r(requirements_path):
    cmd = []
    if shutil.which("pip") is not None:
        cmd = ["pip"]
    elif shutil.which("uv") is not None:
        cmd = ["uv", "pip"]
    else:
        raise RuntimeError("pip not found, failed to install model requirements")
    cmd += ["install", "-r", requirements_path]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
