Contributing code
=================

If you intend to work on this repo and make some changes, please can you let the 
group know what you’re going to work on before you start. This will hopefully 
prevent multiple people working on the same thing, making the merge process a 
bit easier.

How to contribute
-----------------
The preferred way to contribute to this package is to **fork** the 
[main repository](https://github.com/ucl-pond/kde_ebm) on
GitHub:

1. Fork the [project repository](https://github.com/ucl-pond/kde_ebm):
   click on the 'Fork' button at the top right of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone https://github.com/YourGitHubLogin/kde_ebm.git
          [enter username and password as requested]
          $ cd kde_ebm

3. **Create a branch** to hold your changes:

          $ git checkout -b my-feature

   and start making changes. **Never work in the ``master`` branch!**

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-feature

These steps can also be recreated using a Git GUI, for example 
[GitHub Desktop](https://desktop.github.com/).
