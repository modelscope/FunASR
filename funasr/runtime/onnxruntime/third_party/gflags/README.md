[![Build Status](https://travis-ci.org/gflags/gflags.svg?branch=master)](https://travis-ci.org/gflags/gflags)
[![Build status](https://ci.appveyor.com/api/projects/status/4ctod566ysraus74/branch/master?svg=true)](https://ci.appveyor.com/project/schuhschuh/gflags/branch/master)

The documentation of the gflags library is available online at https://gflags.github.io/gflags/.


11 November 2018
----------------

I've just released gflags 2.2.2.

This maintenance release improves lives of Bazel users (no more "config.h" leaking into global include paths),
fixes build with recent MinGW versions, and silences a number of static code analyzer and compiler warnings.
The build targets exported by the CMake configuration of this library are now also prefixed by the package
name "gflags::" following a more recent (unwritten) CMake convention. The unprefixed target names are still
supported to avoid that dependent projects have to be modified due to this change in imported target names.

Please report any further issues with this release using the GitHub issue tracker.


11 July 2017
------------

I've just released gflags 2.2.1.

This maintenance release primarily fixes build issues on Windows and
false alarms reported by static code analyzers.

Please report any further issues with this release using the GitHub issue tracker.


25 November 2016
----------------

I've finally released gflags 2.2.0.

This release adds support for use of the gflags library as external dependency
not only in projects using CMake, but also [Bazel](https://bazel.build/),
or [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/).
One new minor feature is added in this release: when a command flag argument
contains dashes, these are implicitly converted to underscores.
This is to allow those used to separate words of the flag name by dashes
to do so, while the flag variable names are required to use underscores.

Memory leaks reported by valgrind should be resolved by this release.
This release fixes build errors with MS Visual Studio 2015.

Please report any further issues with this release using the GitHub issue tracker.


24 March 2015
-------------

I've just released gflags 2.1.2.

This release completes the namespace change fixes. In particular,
it restores binary ABI compatibility with release version 2.0.
The deprecated "google" namespace is by default still kept as
primary namespace while symbols are imported into the new "gflags" namespace.
This can be overridden using the CMake variable GFLAGS_NAMESPACE.

Other fixes of the build configuration are related to the (patched)
CMake modules FindThreads.cmake and CheckTypeSize.cmake. These have
been removed and instead the C language is enabled again even though
gflags is written in C++ only.

This release also marks the complete move of the gflags project
from Google Code to GitHub. Email addresses of original issue
reporters got lost in the process. Given the age of most issue reports,
this should be negligable.

Please report any further issues using the GitHub issue tracker.


30 March 2014
-------------

I've just released gflags 2.1.1.

This release fixes a few bugs in the configuration of gflags\_declare.h
and adds a separate GFLAGS\_INCLUDE\_DIR CMake variable to the build configuration.
Setting GFLAGS\_NAMESPACE to "google" no longer changes also the include
path of the public header files. This allows the use of the library with
other Google projects such as glog which still use the deprecated "google"
namespace for the gflags library, but include it as "gflags/gflags.h".

20 March 2014
-------------

I've just released gflags 2.1.

The major changes are the use of CMake for the build configuration instead
of the autotools and packaging support through CPack. The default namespace
of all C++ symbols is now "gflags" instead of "google". This can be
configured via the GFLAGS\_NAMESPACE variable.

This release compiles with all major compilers without warnings and passed
the unit tests on  Ubuntu 12.04, Windows 7 (Visual Studio 2008 and 2010,
Cygwin, MinGW), and Mac OS X (Xcode 5.1).

The SVN repository on Google Code is now frozen and replaced by a Git
repository such that it can be used as Git submodule by projects. The main
hosting of this project remains at Google Code. Thanks to the distributed
character of Git, I can push (and pull) changes from both GitHub and Google Code
in order to keep the two public repositories in sync.
When fixing an issue for a pull request through either of these hosting
platforms, please reference the issue number as
[described here](https://code.google.com/p/support/wiki/IssueTracker#Integration_with_version_control).
For the further development, I am following the
[Git branching model](http://nvie.com/posts/a-successful-git-branching-model/)
with feature branch names prefixed by "feature/" and bugfix branch names
prefixed by "bugfix/", respectively.

Binary and source [packages](https://github.com/schuhschuh/gflags/releases) are available on GitHub.


14 January 2014
---------------

The migration of the build system to CMake is almost complete.
What remains to be done is rewriting the tests in Python such they can be
executed on non-Unix platforms and splitting them up into separate CTest tests.
Though merging these changes into the master branch yet remains to be done,
it is recommended to already start using the
[cmake-migration](https://github.com/schuhschuh/gflags/tree/cmake-migration) branch.


20 April 2013
-------------

More than a year has past since I (Andreas) took over the maintenance for
`gflags`. Only few minor changes have been made since then, much to my regret.
To get more involved and stimulate participation in the further
development of the library, I moved the project source code today to
[GitHub](https://github.com/schuhschuh/gflags).
I believe that the strengths of [Git](http://git-scm.com/) will allow for better community collaboration
as well as ease the integration of changes made by others. I encourage everyone
who would like to contribute to send me pull requests.
Git's lightweight feature branches will also provide the right tool for more
radical changes which should only be merged back into the master branch
after these are complete and implement the desired behavior.

The SVN repository remains accessible at Google Code and I will keep the
master branch of the Git repository hosted at GitHub and the trunk of the
Subversion repository synchronized. Initially, I was going to simply switch the
Google Code project to Git, but in this case the SVN repository would be
frozen and force everyone who would like the latest development changes to
use Git as well. Therefore I decided to host the public Git repository at GitHub
instead.

Please continue to report any issues with gflags on Google Code. The GitHub project will
only be used to host the Git repository.

One major change of the project structure I have in mind for the next weeks
is the migration from autotools to [CMake](http://www.cmake.org/).
Check out the (unstable!)
[cmake-migration](https://github.com/schuhschuh/gflags/tree/cmake-migration)
branch on GitHub for details.


25 January 2012
---------------

I've just released gflags 2.0.

The `google-gflags` project has been renamed to `gflags`.  I
(csilvers) am stepping down as maintainer, to be replaced by Andreas
Schuh.  Welcome to the team, Andreas!  I've seen the energy you have
around gflags and the ideas you have for the project going forward,
and look forward to having you on the team.

I bumped the major version number up to 2 to reflect the new community
ownership of the project.  All the [changes](ChangeLog.txt)
are related to the renaming.  There are no functional changes from
gflags 1.7.  In particular, I've kept the code in the namespace
`google`, though in a future version it should be renamed to `gflags`.
I've also kept the `/usr/local/include/google/` subdirectory as
synonym of `/usr/local/include/gflags/`, though the former name has
been obsolete for some time now.


18 January 2011
---------------

The `google-gflags` Google Code page has been renamed to
`gflags`, in preparation for the project being renamed to
`gflags`.  In the coming weeks, I'll be stepping down as
maintainer for the gflags project, and as part of that Google is
relinquishing ownership of the project; it will now be entirely
community run.  The name change reflects that shift.


20 December 2011
----------------

I've just released gflags 1.7.  This is a minor release; the major
change is that `CommandLineFlagInfo` now exports the address in memory
where the flag is located.  There has also been a bugfix involving
very long --help strings, and some other minor [changes](ChangeLog.txt).

29 July 2011
------------

I've just released gflags 1.6.  The major new feature in this release
is support for setting version info, so that --version does something
useful.

One minor change has required bumping the library number:
`ReparseCommandlineFlags` now returns `void` instead of `int` (the int
return value was always meaningless).  Though I doubt anyone ever used
this (meaningless) return value, technically it's a change to the ABI
that requires a version bump.  A bit sad.

There's also a procedural change with this release: I've changed the
internal tools used to integrate Google-supplied patches for gflags
into the opensource release.  These new tools should result in more
frequent updates with better change descriptions.  They will also
result in future `ChangeLog` entries being much more verbose (for better
or for worse).

See the [ChangeLog](ChangeLog.txt) for a full list of changes for this release.

24 January 2011
---------------

I've just released gflags 1.5.  This release has only minor changes
from 1.4, including some slightly better reporting in --help, and
an new memory-cleanup function that can help when running gflags-using
libraries under valgrind.  The major change is to fix up the macros
(`DEFINE_bool` and the like) to work more reliably inside namespaces.

If you have not had a problem with these macros, and don't need any of
the other changes described, there is no need to upgrade.  See the
[ChangeLog](ChangeLog.txt) for a full list of changes for this release.

11 October 2010
---------------

I've just released gflags 1.4.  This release has only minor changes
from 1.3, including some documentation tweaks and some work to make
the library smaller.  If 1.3 is working well for you, there's no
particular reason to upgrade.

4 January 2010
--------------

I've just released gflags 1.3.  gflags now compiles under MSVC, and
all tests pass.  I **really** never thought non-unix-y Windows folks
would want gflags, but at least some of them do.

The major news, though, is that I've separated out the python package
into its own library, [python-gflags](http://code.google.com/p/python-gflags).
If you're interested in the Python version of gflags, that's the place to
get it now.

10 September 2009
-----------------

I've just released gflags 1.2.  The major change from gflags 1.1 is it
now compiles under MinGW (as well as cygwin), and all tests pass.  I
never thought Windows folks would want unix-style command-line flags,
since they're so different from the Windows style, but I guess I was
wrong!

The other changes are minor, such as support for --htmlxml in the
python version of gflags.

15 April 2009
-------------

I've just released gflags 1.1.  It has only minor changes fdrom gflags
1.0 (see the [ChangeLog](ChangeLog.txt) for details).
The major change is that I moved to a new system for creating .deb and .rpm files.
This allows me to create x86\_64 deb and rpm files.

In the process of moving to this new system, I noticed an
inconsistency: the tar.gz and .rpm files created libraries named
libgflags.so, but the deb file created libgoogle-gflags.so.  I have
fixed the deb file to create libraries like the others.  I'm no expert
in debian packaging, but I believe this has caused the package name to
change as well.  Please let me know (at
[[mailto:google-gflags@googlegroups.com](mailto:google-gflags@googlegroups.com)
google-gflags@googlegroups.com]) if this causes problems for you --
especially if you know of a fix!  I would be happy to change the deb
packages to add symlinks from the old library name to the new
(libgoogle-gflags.so -> libgflags.so), but that is beyond my knowledge
of how to make .debs.

If you've tried to install a .rpm or .deb and it doesn't work for you,
let me know.  I'm excited to finally have 64-bit package files, but
there may still be some wrinkles in the new system to iron out.

1 October 2008
--------------

gflags 1.0rc2 was out for a few weeks without any issues, so gflags
1.0 is now released.  This is much like gflags 0.9.  The major change
is that the .h files have been moved from `/usr/include/google` to
`/usr/include/gflags`.  While I have backwards-compatibility
forwarding headeds in place, please rewrite existing code to say
```
   #include <gflags/gflags.h>
```
instead of
```
   #include <google/gflags.h>
```

I've kept the default namespace to google.  You can still change with
with the appropriate flag to the configure script (`./configure
--help` to see the flags).  If you have feedback as to whether the
default namespace should change to gflags, which would be a
non-backwards-compatible change, send mail to
`google-gflags@googlegroups.com`!

Version 1.0 also has some neat new features, like support for bash
commandline-completion of help flags.  See the [ChangeLog](ChangeLog.txt)
for more details.

If I don't hear any bad news for a few weeks, I'll release 1.0-final.
