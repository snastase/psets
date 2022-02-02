# Problem sets for NEU502b

### First-time setup
Log into your GitHub account, navigate to the [2022-NEU502b organization](https://github.com/2022-NEU502b), and click on the [`psets` repository](https://github.com/2022-NEU502b/psets).

Click on the _Fork_ button at top right and select your own GitHub account. This "fork" operation creates a copy of the original `psets` repository on your personal GitHub account.

Now, navigate to _your fork_ of the `psets` repository on your own account (e.g. https://github.com/username/psets; you may be automatically redirected to your fork). Click on the green _Code_ button and copy the URL for the repository ending in `.git`.

Open a terminal and navigate (`cd`) to a location on your personal computer or the server where you want to work on the problem sets (activate your conda environment as well; e.g. `conda activate neu502b`). We recommend navigating to the directory that contains the `demos` directory (but not inside the `demos` directory!). For example, if your `demos` directory is at the following location `~/neu502b/demos`, navigate to the `~/neu502b` directory. We'll place the `psets` repository side-by-side to the `demos` directory.

Finally, we'll "clone" _your fork_ of the `psets` repository:
```
cd ~/neu502b
git clone https://github.com/username/psets.git
```

### Working on the problem set
First, navigate into the `psets/fmri-ps1` directory and create a copy of the exercises notebook with your name in the filename:
```
cd ~/neu502b/psets/fmri-ps1
cp fmri-ps1-exercises.ipynb yourname-fmri-ps1-exercises.ipynb
```

Now you can start to make and save changes in the new copy of the exercises notebook. When you've made some progress or finished the exercises, you'll use the typical Git workflow to `add` your changes, `commit` them, and ultimately `push` them to your fork of the repository on GitHub. Check the status of your repository before use Git commands.
```
git status
git add yourname-fmri-ps1-exercises.ipynb
git commit -m "Finished problem set 1"
```

When you're finished and ready to submit the problem set, you'll make a "pull request" to submit your notebook to the original `psets` repository in the [2022-NEU502b organization](https://github.com/2022-NEU502b). Make sure you've committed all your changes and pushed them to _your fork_ of the repository on GitHub. Open your browser and navigate to your fork of the repository on GitHub. Click the _Make a pull request_ button.
