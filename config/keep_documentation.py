import os

working_directory = r"./"
retain = ["config", ".github", ".git", "testASDDSA", "docs", ".venv"]
os.chdir(working_directory)

for item in next(os.walk(working_directory))[1]:
    if item not in retain:
        for path, subdirs, files in os.walk(item):
            for name in files:
                path_file = os.path.join(path, name)
                os.remove(path_file)
                os.system(f'git rm {path_file}')

for path_file in [f for f in os.listdir('.') if os.path.isfile(f)]:
    os.remove(path_file)
    os.system(f'git rm {path_file}')
