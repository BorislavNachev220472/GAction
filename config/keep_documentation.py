import os
import shutil

working_directory = r"./"
retain = ["config", ".github", ".git", "src"]
os.chdir(working_directory)

for item in next(os.walk(working_directory))[1]:
    if item not in retain:
        shutil.rmtree(item)

