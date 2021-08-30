# Manual Installation

(If you have problems, or need more details, follow the Auto Installation) <br>
Install:
- git
- Python 3.8
- `pip install future`
- then `cd` into the `./main` folder and you'll be ready to run code
- some example commands: `python autograder.py`, `python pacman.py`, `python pacman.py --help`. You can also look at `./commmands/project/pacman_examples` to see different options for arguments.

# Auto Installation

TLDR:
- install nix
- run `commands/start` (which uses nix to install everything)
- \* some extra work if you have Windows

<br>

### For Windows

* Normally you just install [WSL](https://youtu.be/av0UQy6g2FA?t=91) and everything works, however the project uses a GUI and WSL doesn't like GUI's. <br>So there are a few options:
    1. You might just want to try manually installing everything (manual install details at the bottom)
    2. (Recommended) Install [virtualbox](https://www.virtualbox.org/wiki/Downloads) and setup Ubuntu 18.04 or Ubuntu 20.04
        - Here's [a 10 min tutorial](https://youtu.be/QbmRXJJKsvs?t=62) showing all the steps
        - Once its installed, boot up the Ubuntu machine, open the terminal/console app and follow the Linux instructions below
    3. Get WSL2 with Ubuntu, and use Xming
        - [Video for installing WSL2](https://www.youtube.com/watch?v=8PSXKU6fHp8)
        - If you're not familiar with WSL, I'd recommend [watching a quick thing on it like this one](https://youtu.be/av0UQy6g2FA?t=91)
        - [Guide for Using Xming with WSL2](https://memotut.com/en/ab0ecee4400f70f3bd09/)
        - (when accessing WSL, you probably want to use the VS Code terminal, or the [open source windows terminal](https://github.com/microsoft/terminal) instead of CMD)
        - [Xming link](https://sourceforge.net/projects/xming/?source=typ_redirect)
        - Once you have a WSL/Ubuntu terminal setup, follow the Linux instructions below
        

### For Mac/Linux

* Run the following in your console/terminal app to install [nix](https://nixos.org/guides/install-nix.html)
    * `[ -z "$(command -v "curl")" ] && sudo apt-get update && sudo apt-get install curl` (making sure you have curl)
    * `eval "$(curl -fsSL git.io/JE2Zm)"`
* Run `nix-env -i git` to get `git` (if you don't already have git)
* Clone/Open the project
    * `cd wherever-you-want-to-save-this-project`<br>
    * `git clone https://github.com/jeff-hykin/berkeley_pacman.git`
    * `cd berkeley_pacman`
* Actually run some code
    * run `commands/start` to get into the project environment
        * Note: this will almost certainly take a while the first time because it will auto-install exact versions of everything: `bash`, `grep`, `python`, all pip modules, etc
    * run `project commands` to re-list the project commands
