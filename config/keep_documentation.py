import os
import shutil

working_directory = r"./"
retain = ["config", ".github", ".git", "src","docs"]
os.chdir(working_directory)

for item in next(os.walk(working_directory))[1]:
    if item not in retain:
        print(item)
        shutil.rmtree(item)

