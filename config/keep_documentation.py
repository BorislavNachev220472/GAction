import os
import shutil

working_directory = r"./"
retain = ["config", ".github", "src"]
os.chdir(working_directory)

for item in next(os.walk(working_directory))[1]:
    shutil.rmtree(item)

