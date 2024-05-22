import subprocess


def install_requirements(requirements_path):
    try:
        result = subprocess.run(
            ["pip", "install", "-r", requirements_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # check status
        if result.returncode == 0:
            print("install model requirements successfully")
            return True
        else:
            print("fail to install model requirements! ")
            print("error", result.stderr)
            return False
    except Exception as e:
        result = subprocess.run(
            ["pip", "install", "-r", requirements_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # check status
        if result.returncode == 0:
            print("install model requirements successfully")
            return True
        else:
            print("fail to install model requirements! ")
            print("error", result.stderr)
            return False
