import os


working_directory = r"./"
retain = ["config", ".github", ".git", "src"]
os.chdir(working_directory)

for item in os.listdir(os.getcwd()):
    if item not in retain:
        os.remove(item)
