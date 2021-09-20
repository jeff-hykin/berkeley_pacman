# Manual Setup (No Project Environment)

(If you have problems, or need more details, follow the Auto Installation) <br>
Install:
- git
- Python 3.8
- `pip install future`
- then `cd` into the `./main` folder and you'll be ready to run code
- some example commands: `python autograder.py`, `python pacman.py`, `python pacman.py --help`. You can also look at `./commmands/project/pacman_examples` to see different options for arguments.


# Project Environment (Alternative to Manual Setup)

### If you're an Experienced/Senior Dev

- (Don't git clone)
- Run this: `repo=https://github.com/jeff-hykin/berkeley_pacman eval "$(curl -fsSL git.io/JE2Zm || wget -qO- git.io/JE2Zm)"`
- If you're on Windows, run it inside WSL (Ubuntu 20.04 preferably)
- If you're a responsible human being and therefore don't want run a sketchy internet script, props to you 👍. Take a look at the explaination below and you'll be able to run the commands yourself.

### If the above instructions didn't make sense

- Mac/Linux users
    - open up your terminal/console app
    - use `cd` to get to the folder where you want this project ([tutorial on how to use cd here](https://github.com/jeff-hykin/fornix/blob/b6fd3313beda4f80b7051211cb790a4f34da590a/documentation/images/cd_tutorial.gif))
    - (If you get errors on the next step -> keep reading)
    - Type this inside your terminal/console <br>`repo=https://github.com/jeff-hykin/berkeley_pacman eval "$(curl -fsSL git.io/JE2Zm || wget -qO- git.io/JE2Zm)"`<br>[press enter]
    - Possible errors:
        - On MacOS, if your hard drive is encrypted on BigSur, you might need to [follow these steps](https://stackoverflow.com/questions/67115985/error-installing-nix-on-macos-catalina-and-big-sur-on-filevault-encrypted-boot-v#comment120393385_67115986)
        - On Linux, if you're running a *really* barebones system that somehow doesn't have either `curl` or `wget`, install curl or wget and rerun the previous step
- Windows users
    - Normally you just install [WSL](https://youtu.be/av0UQy6g2FA?t=91) and follow the Linux instructions, however the project uses a GUI and WSL doesn't like GUI's. <br>So there are a few options:
        1. You might just want to try manually installing everything (manual install details at the top)
        2. (Recommended) Install [virtualbox](https://www.virtualbox.org/wiki/Downloads) and setup Ubuntu 18.04 or Ubuntu 20.04
            - Here's [a 10 min tutorial](https://youtu.be/QbmRXJJKsvs?t=62) showing all the steps
            - Once its installed, boot up the Ubuntu machine, open the terminal/console app and follow the Linux instructions
        3. Get WSL2 with Ubuntu, and use Xming
            - [Video for installing WSL2](https://www.youtube.com/watch?v=8PSXKU6fHp8)
            - If you're not familiar with WSL, I'd recommend [watching a quick thing on it like this one](https://youtu.be/av0UQy6g2FA?t=91)
            - [Guide for Using Xming with WSL2](https://memotut.com/en/ab0ecee4400f70f3bd09/)
            - (when accessing WSL, you probably want to use the VS Code terminal, or the [open source windows terminal](https://github.com/microsoft/terminal) instead of CMD)
            - [Xming link](https://sourceforge.net/projects/xming/?source=typ_redirect)
            - Once you have a WSL/Ubuntu terminal setup, follow the Linux instructions

After you've finished working and close the terminal, you can always return to project environment by doing
- `cd wherever-you-put-the-project`
- `commands/start`


### What is that `eval` command doing?

1. Installing nix [manual install instructions here](https://nixos.org/guides/install-nix.html)
2. Installing `git` (using nix) if you don't already have git
3. It clones the repository
4. It `cd`'s inside of the repo
5. Then it runs `commands/start` to enter the project environment