from glob import glob
import nbformat
from nbclient import NotebookClient
import subprocess


def run_nb(nb):
    with open(nb) as f:
        nb = nbformat.read(f, as_version=4)
        client = NotebookClient(nb)
        client.execute()


py_scripts = glob('examples' + '/**/*.py', recursive=True)
notebooks = glob('examples' + '/**/*.ipynb', recursive=True)

num_tests = len(notebooks) + len(py_scripts)
failed_test = 0

for f in py_scripts:
    print(f'running {f}')
    res = subprocess.run(['python', f])
    if res.returncode != 0:
        failed_test += 1
        print(f'{f} failed')
        continue
    print(f'{f} succeeded')

for f in notebooks:
    print(f'running {f}')
    try:
        run_nb(f)
    except:                     # noqa: E722
        failed_test += 1
        print(f'{f} failed')
        continue
    print(f'{f} succeeded')

print(f'Tests exectuted: {num_tests}')
print(f'Tests failed:    {failed_test}')
print(f'Success rate:    {(num_tests-failed_test)/num_tests * 100} %')

if failed_test > 0:
    exit(1)
