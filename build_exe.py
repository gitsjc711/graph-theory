# build_exe.py
import os
import shutil
import PyInstaller.__main__

# 清理之前的构建
for dir_name in ['build', 'dist', '__pycache__']:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

# 打包参数
params = [
    'main.py',                     # 主程序文件
    '--name=GraphAlgorithmVisualizer',
    '--onefile',                   # 打包为单个exe文件
    '--windowed',                  # 窗口程序，不显示控制台
    '--clean',                     # 清理临时文件
    '--noconfirm',                 # 覆盖现有文件时不提示
    '--add-data=.;.',              # 包含当前目录所有文件
    '--hidden-import=matplotlib.backends.backend_tkagg',
    '--hidden-import=networkx',
    '--hidden-import=PIL',         # 处理图片可能需要的
    '--hidden-import=scipy',       # networkx 可能需要
    '--hidden-import=scipy.sparse.csgraph',
    '--collect-all=matplotlib',    # 包含matplotlib的所有数据
    '--collect-all=networkx',      # 包含networkx的所有数据
]

# 执行打包
PyInstaller.__main__.run(params)

print("打包完成！可执行文件在 dist 文件夹中。")