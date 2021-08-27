# Manual Installation

(If you have problems, follow the Auto installation)
Install:
- git
- Python 3.8
- then `cd` into the `./main` folder and you'll be ready to run code
- `python autograder.py`, `python pacman.py`, `python pacman.py --help` and take a look at `./commmands/project/pacman_examples` to see different options for argument

# Auto Installation

TLDR:
- install nix
- run `commands/start` (which uses nix to install everything)
- \* some extra work if you have Windows

<br>

### For Windows

* Normally you just install [WSL](https://youtu.be/av0UQy6g2FA?t=91) and everything works, however the project uses a GUI and WSL doesn't like GUI's. So there are a few options:
    1. You might just want to try manually installing everything (manual install details at the bottom)
    2. (Recommended) Install [virtualbox](https://www.virtualbox.org/wiki/Downloads) and setup Ubuntu 18.04 or Ubuntu 20.04
        - Here's [a 10 min tutorial](https://youtu.be/QbmRXJJKsvs?t=62) showing all the steps
        - Once its installed, open up the Ubuntu terminal app and follow the linux instructions below
    3. Get WSL2 with Ubuntu, and use Xming
        - [Video for installing WSL2](https://www.youtube.com/watch?v=8PSXKU6fHp8)
        - If you're not familiar with WSL, I'd recommend [watching a quick thing on it like this one](https://youtu.be/av0UQy6g2FA?t=91)
        - [Guide for Using Xming with WSL2](https://memotut.com/en/ab0ecee4400f70f3bd09/)
        - (when accessing WSL, you probably want to use the VS Code terminal, or the [open source windows terminal](https://github.com/microsoft/terminal) instead of CMD)
        - [Xming link](https://sourceforge.net/projects/xming/?source=typ_redirect)
        - Once you have a WSL/Ubuntu terminal setup, follow the linux instructions below
        

### For Mac/Linux

* Install [nix](https://nixos.org/guides/install-nix.html), more detailed guide [here](https://nixos.org/manual/nix/stable/#chap-installation)
    * Just run the following in your console/terminal app
        * `sudo apt-get update 2>/dev/null`
        * If you're on MacOS Big Sur
            * run `sudo diskutil apfs addVolume disk1 APFS 'Nix Store' -mountpoint /nix`
            * then run `sh <(curl -L https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume`
            * (see [here](https://duan.ca/2020/12/13/nix-on-macos-11-big-sur/) if you have issues and need more details) 
        * If you're on MacOS Catalina, run:
            * `sh <(curl -L https://nixos.org/nix/install) --darwin-use-unencrypted-nix-store-volume `
        * If you're on linux or and older MacOS, run:
            * `curl -L https://nixos.org/nix/install | bash`
        * `source $HOME/.nix-profile/etc/profile.d/nix.sh`
        * (may need to restart console/terminal)
* Install `git`
    * (if you don't have git just run `nix-env -i git`)
* Clone/Open the project
    * `cd wherever-you-want-to-save-this-project`<br>
    * `git clone https://github.com/*this-repo-url*`
    * `cd *this-repo*`
* Actually run some code
    * run `commands/start` to get into the project environment
        * Note: this will almost certainly take a while the first time because it will auto-install exact versions of everything: `bash`, `grep`, `python`, all pip modules, etc
    * run `project commands` to list the project commands
