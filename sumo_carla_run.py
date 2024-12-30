# -*- coding: utf-8 -*-


import subprocess
import os


def run_command_in_directory(command, directory):
    try:
        # 保存当前工作目录
        original_dir = os.getcwd()

        # 切换到目标目录
        os.chdir(directory)

        # 运行命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # 输出命令执行结果
        if result.stdout:
            print("命令输出:")
            print(result.stdout)
        if result.stderr:
            print("命令错误信息:")
            print(result.stderr)

        # 返回运行结果
        return result.returncode
    finally:
        # 确保返回原始目录
        os.chdir(original_dir)


# 指定目录和命令
directory = r"F:\carla\Co-Simulation\Sumo"
command = r"python.exe .\run_synchronization.py .\examples\town_L_3.sumocfg --sumo-gui"

# 运行命令
return_code = run_command_in_directory(command, directory)
print(f"命令返回码: {return_code}")
