# How to test your code on GitHub

<hr>

## Organise your code sensibly


Consider this script - a GLM regression using Ordinary Least Squares (OLS):

```python
#!/usr/bin/env python

import numpy as np

data  = np.loadtxt('data.txt')
model = np.loadtxt('design.txt')

fit   = np.linalg.inv(model.T @ model) @ model.T @ data
error = data - (model @ fit)

np.savetxt('pes.txt', fit)
np.savetxt('residuals.txt', error)
```

How would you validate that this code is doing what it is supposed to? You'd have to:

1. Create some input files - `data.txt` and `design.txt`.
2. Run the script, probably from another Python/bash script, or directly from a terminal.
3. Load/evaluate the output files - `pes.txt` and `residuals.txt`.

And you'd have to do this for every test you wanted to perform.


The structure of this code makes testing it awkward:

 - I/O (reading input files/saving output files) is tightly coupled to the core logic (the regression).
 - None of the logic can be programmatically called - the entire script needs to be run on every test.

Now consider this code:


```python
#!/usr/bin/env python

import numpy as np
import sys

def ols(data, model):
    """Perform ordinary-least-squares regression. """
    fit   = np.linalg.inv(model.T @ model) @ model.T @ data
    error = data - (model @ fit)
    return fit, error

def main(args=None):
    """Fit a linear model to some data. """

    if args is None:
        args = sys.argv[1:]

    datafile   = args[0]
    designfile = args[1]
    fitfile    = args[2]
    errfile    = args[3]

    # load data and model
    data  = np.loadtxt(datafile)
    model = np.loadtxt(designfile)

    # perform OLS regression
    fit, error = ols(data, model)

    # save parameter estimates and residuals
    np.savetxt(fitfile, fit)
    np.savetxt(errfile, error)

if __name__ == '__main__':
    sys.exit(main())
```

This is a better design:

 - The core logic (OLS regression) is in a standalone function (`ols`).
 - The I/O logic is in another function (`main`), that can be programmatically called.
 - Both of these functions can be called directly, without having to run the entire script from a separate process.


<hr>

### General advice for code organisation


> [!TIP]
> Wherever possible, strive to write functions which are:\
\
&nbsp;- Small\
&nbsp;- Simple\
&nbsp;- Self-contained\
&nbsp;- Without side-effects\
\
They will be easier to understand, easier to re-use, and easier to test.


> [!TIP] 
> Try to separate procedural operations (e.g. loading/saving data) from purely functional/numeric operations. Doing so will make the critical parts of your code easier to test.


> [!TIP] 
> Try to structure your code so that it can be run on small "toy" datasets - this will allow you to test your code quickly and locally (i.e. on your laptop).


> [!TIP] 
> Use functions, classes, modules, and packages to arrange your project in a sensible manner.\
\
There is no single answer to the question of how you should organise your code. If it is easy to understand, navigate, and test (and it does its job), then it is a good design.


<hr>

## Write code to test your code


How would you test the code above? You can simply write a couple of functions which call the `ols` and `main` functions directly, e.g.:

```python

test_model = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).T
test_data  = (test_model[:, 0] * 5) + (test_model[:, 1] * 20)

def test_ols():
    fit, err = ols(test_data, test_model)

    assert np.isclose(fit, [5, 20]).all()
    assert np.isclose(err, 0)      .all()

def test_main():
    np.savetxt('data.txt', test_data)
    np.savetxt('design.txt', test_model)

    main(('data.txt', 'design.txt', 'fit.txt', 'err.txt'))

    fit = np.loadtxt('out_fit.txt')
    err = np.loadtxt('out_err.txt')

    assert np.isclose(fit, [5, 20]).all()
    assert np.isclose(err, 0)      .all()
```


Then you can use a testing framework such as [`pytest`](https://docs.pytest.org/en/stable/), which will automatically run all of your test functions (it scans your code and runs every function starting with `test_`):

```
$ pytest
================= test session starts ================

my_ols/tests/test_ols.py::test_ols PASSED       [ 50%]
my_ols/tests/test_ols.py::test_main PASSED      [100%]

================== 2 passed in 0.06s =================
```

You can combine `pytest` with a companion tool [`coverage`](https://coverage.readthedocs.io/en/7.13.5/) to generate a report telling you how much of your code was covered by the tests:

```
Name                 Stmts   Miss  Cover
----------------------------------------
my_ols/__init__.py       0      0   100%
my_ols/main.py          20      2    90%
----------------------------------------
TOTAL                   20      2    90%
```

`coverage` can also produce a report telling you exactly which parts of your code were and weren't covered by your tests - run `pytest --cov-report=html`, then open `htmlcov/index.html` in a web browser.

> [!NOTE]
> Regardless of which programming language you are working in, there is a test framework available for you to use - here are just a few examples:
>
>   - C++: https://github.com/catchorg/Catch2/
>   - MATLAB: https://www.mathworks.com/help/matlab/matlab-unit-test-framework.html
>   - Javascript: https://jestjs.io/
>   - Rust: https://doc.rust-lang.org/rust-by-example/testing/unit_testing.html
>   - Go: https://go.dev/doc/tutorial/add-a-test


<hr>

### Where to store your tests


You have several options when deciding where to store your tests within your project:

You can put your tests in a separate directory from your code, e.g.:

```
my_ols/
    __init__.py
    main.py
tests/
    test_ols.py
```

You can put your tests in the same directory, e.g.:

```
my_ols/
    __init__.py
    main.py
    test_ols.py
```

You can put your tests in a sub-directory/sub-package, e.g.:

```
my_ols/
    __init__.py
    main.py
    tests/
        __init__.py
        test_ols.py
```

This decision is often a matter of personal preference. I personally prefer the last option (storing tests in a sub-directory/sub-package), as it makes it easier to re-use your test code whilst keeping it separate from your normal code.

Whatever option you choose, `pytest` is usually pretty good at automatically finding your test functions wherever they are stored.


<hr>

### General advice for writing tests

> [!TIP]
> Write and run tests at the same time as you are writing the code. Doing so will help you to better understand and trust your code.


> [!TIP] 
> Write tests that run quickly - if your tests only take a few seconds to run, you are much more likely to run them regularly.\
\
With `pytest` you can run a subset of tests, e.g. `pytest -k test_ols` will only run the `test_ols` function in the above example.

> [!TIP]
> Write tests for individual functions - these are known as _unit tests_. For example, `test_ols` above is an example of a unit test.

> [!TIP]
> Write tests for the entire program - these are known as _integration tests_. `test_main` above is an example of an integration test.

> [!TIP]
> Whenever you fix a bug in your code, write a test for it, so you’ll know if it is ever accidentally re-introduced. This is known as a _regression test_.

> [!TIP]
> If you are releasing your software (e.g. to [PyPi](https://pypi.org/) or [conda-forge](https://conda-forge.org/)), write tests for basic functionality (e.g. checking that `my_program --help` doesn't crash, `my_program --version` reports the correct version, etc.).\
\
These are known as _smoke tests_, and help to avoid embarrassment from silly mistakes.

> [!TIP]
> Don’t be afraid to refactor and rearrange your code as it evolves.  If you have a test suite, then you can use it to ensure that you are
not breaking your code during a refactor.\
\
If you use version control, then you can experiment freely with your design, and easily revert back to a known good version if needed.


<hr>

### Use dependency injection to test awkward code


Not all code is easy to test. You might be depending on code written by somebody else, or may not have the time to re-organise your own code. Imagine that you have some code which looks like this:


```python
import requests

def load_parameters():
    resp   = requests.get('https://www.config.org/parameters.txt')
    config = resp.text.split('\n')
    param1 = int(config[0])
    param2 = int(config[1])
    param3 = int(config[2])
    return param1, param2, param3

def run_preprocessing(data):
    p1, p2, p3 = load_parameters()
    return data * p1 * p2 / p3
```


The behaviour of the `run_preprocessing` function depends on the values that it retrieves from an external source over which you have no control. How would you test this function?

> [!IMPORTANT]
> Remember the first piece of advice [given above](#general-advice-for-code-organisation) - _strive to write functions which are small, simple, self-contained, and without side-effects_. This code is **not** self-contained. Try to avoid writing this sort of code - a better design would be to pass all required input parameters as arguments, and push the responsibility of downloading the parameters out to the calling code.


If you are coding in Python, you're in luck - the built-in [`unittest.mock`](https://docs.python.org/3/library/unittest.mock.html) library makes testing this code easy:


```python
from unittest import mock

import my_ols.preprocess as preproc

def test_run_preprocessing():

    with mock.patch('my_ols.preprocess.load_parameters', return_value=(10, 10, 10)):
        assert preproc.run_preprocessing(10) == 100

    with mock.patch('my_ols.preprocess.load_parameters', return_value=(1, 2, 5)):
        assert preproc.run_preprocessing(10) == 4
```


<hr>

### Simplify your tests by making them data-driven


The test code above is fine for testing a couple of scenarios, but it might become unwieldy if you want to test dozens or hundreds of inputs. Think about how to structure your test code to minimise the amount of boilerplate code that you have to write. For example, for this test we can just put all of our inputs and expected results into a list which we can loop over:


```python
from unittest import mock

import my_ols.preprocess as preproc

def test_run_preprocessing():

    # Each test is a tuple of (input data, input parameters, expected result)
    tests = [
        (10, (10, 10, 10), 100),
        (10, (1, 2, 4), 5),
        (20, (1, 2, 5), 8)
    ]

    for data, params, result in tests:
        with mock.patch('my_ols.preprocess.load_parameters', return_value=params):
            assert preproc.run_preprocessing(data) == result
```

Now it is much easier to add as many tests cases as you need.


<hr>

## Run your tests automatically using CI / CD


_Continuous Integration_ and _Continuous Deployment_ are terms used in software engineering which simply refer to the idea of automating commonly performed tasks. Instead of manually running your tests after every change, you can set things up so that your tests run automatically when you push changes to your code repository.

> [!NOTE]
> You can also automatically build and release your code to PyPi, build and publish your documentation, and more generally accomplish pretty much anything that can be scripted.

The way that CI / CD works is roughly:

1. You push some changes to your code repository (e.g. GitHub).
2. GitHub notifies your CI provider about the changes.
3. The CI provider starts up a Docker container or a virtual machine running somewhere in the cloud.
4. Your code is downloaded to the container/VM, then a script (which you have written) runs your tests.

If you are using GitHub, then the CI infrastructure is also provided by GitHub (known as _GitHub Actions_). But you can set up an external CI provider if you wish, such as [Travis](https://www.travis-ci.com/), [Circle CI](https://circleci.com/) or [Microsoft Azure](https://azure.microsoft.com/) (there are many to choose from, some free, some paid).

We are now going to walk through how to set up automated testing on a simple Python project on GitHub.  We are going to use the `Research code test` project (https://github.kcl.ac.uk/k2258483/my_ols), but hopefully you will learn enough to be able to set up your own project later on.

<hr>

### Set up a local development environment

> [!WARNING]
> You will need a GitHub account to complete the following sections, and will need to have SSH key-based authentication configured. Follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) if you haven't already done so.

First of all let's set up your local machine so that you can hack on the code and run the tests locally.

To accomplish this, you will need to create a Python environment with the project dependencies  installed, along with `pytest` and `coverage`. You can do this in any way you wish (e.g. `conda`, `micromamba`, `python -m venv`, etc). but if you are unsure, we recommend using a [tool called `uv`](https://docs.astral.sh/uv/). If you don't already have `uv` installed, follow these steps:

1. Open a terminal and run this command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Close the terminal and open a new one, and run:
   ```bash
   uv --version
   ```
   If the installation worked, you should see something like:
   ```
   uv 0.11.11 (ed7b06001 2026-05-06 aarch64-apple-darwin)
   ```

Now let's get a copy of the code:

1. Clone the GitHub repository so you have a local copy:
   ```bash
   git clone https://github.com/rbesenczi/my_ols.git
   ```
   Or, if you have set up SSH key-based authentication (meaning you won't have to enter your username/password):
   ```bash
   git clone git@github.com:rbesenczi/my_ols.git
   ```

2. Change into the `my_ols` directory, and create a Python environment with the project dependencies. If you are using `uv`, run:
   ```bash
   uv sync --all-extras
   ```

3. Now you can run the `my_ols` command:
   ```bash
   uv run my_ols
   ```
   You should see a very minimal usage description:
   ```
   Usage: my_ols.main data_in design_in fit_out error_out
   ```

4. Run the unit tests with this command:
   ```bash
   uv run pytest
   ```
   Hopefully all of the tests passed!

You now have a local development environment for `my_ols`, and can run the test suite manually whenever you need. Now let's set things up so those tests are executed automatically by GitHub.

> [!NOTE]
> The `uv` command takes care of creating and activating Python environments for you - if you are using `conda` or `python -m venv`, you always need to make sure that you have activated your environment before calling `python`, `pip`, or (in this case) `pytest` . In contrast, when you use  `uv run <command>` it ensures that the correct environment is used.


<hr>

## Set up CI / CD on GitHub


One advantage of using GitHub is that you don't need to set up your own CI infrastructure - GitHub will run your tests on its own computing infrastructure. This is known as _GitHub Actions_, which is free for small projects.

> [!NOTE]
> GitHub Actions may not be free forever, so you should always be prepared to switch to a different CI provider.


You need to define one or more **workflows**. Each worfklow is contained in its own `yaml` file, within a `.github/worfklows/` directory in your repository. Within this file you must specify the conditions under which the workflow will run, and the individual jobs that make up the workflow.


We are just going to create a single workflow which runs our tests whenever new commits are pushed to the `main` branch.


1. Log into your GitHub account, and create a new empty repository called `my_ols`:

2. Open a terminal, change into your `my_ols` directory, and run the following commands (replace `<username>` with your GitHub username):
   ```bash
    git remote add origin https://github.kcl.ac.uk/<username>/my_ols.git
    git branch -M main
    git push -u origin main
   ```

3. Copy and paste the following commands - this will create a new file `.github/worfklows/test.yaml`, containing the GitHub actions configuration:
   ```bash
   mkdir -p .github/workflows/
   cat << EOF > .github/workflows/test.yaml
   on:
     push:
       branches: main

   jobs:
     test:
       runs-on: ubuntu-latest

       steps:
         - uses: actions/checkout@v4

         - name: Set up Python 3.14
           uses: actions/setup-python@v6
           with:
             python-version: "3.14"

         - name: Install package and dependencies
           run: pip install ".[test]"

         - name: Run tests
           run: pytest
   EOF
   ```

4. Commit and push these changes to your repository.

5. Open `https://github.com/<username>/my_ols/actions` in a web browser (replace `<username>` with your username), and click through to watch your tests run!


Let's look at our GitHub Actions configuration in more detail. The first section tells GitHub when our workflow should be executed - whenever commits are pushed to the `main` branch:
```yaml
on:
  push:
    branches: main
```

The next section defines our jobs - in this example, we just have a single job named `test`. GitHub Actions uses virtual machines (VMs) instead of Docker images, so here we have specified `ubuntu-latest` (it is also possible to run jobs on windows and macOS):
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
```

Each job contains a list of `steps` - GitHub will execute each step in sequence:

1. The first step simply checks out your repository. This step uses a "pre-canned" recipe called `actions/checkout`:
   ```yaml
   - uses: actions/checkout@v4
   ```
   There are many pre-canned steps that you can re-use within a GitHub Actions workflow - you can browse a wide range at the [GitHub Marketplace](https://github.com/marketplace?type=actions).

2. The next step installs a Python environment, again using a pre-canned recipe called `actions/setup-python`:
   ```yaml
   - name: Set up Python 3.14
     uses: actions/setup-python@v6
     with:
       python-version: "3.14"
   ```

3. The next two steps install our package and run the tests, in the same way that we used in our GitLab CI / CD configuration:
   ```yaml
   - name: Install package and dependencies
     run: pip install ".[test]"

   - name: Run tests
     run: pytest
   ```

> [!NOTE]
> Workflow syntax for GitHub Actions: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax

# Towards a more robust software process - Issue Tracking with GitHub Issues

GitHub Issues provides a convenient way to record bugs, feature requests, improvements, and other development tasks alongside the source code they affect. When Issues are combined with Git branches and pull requests, you can track a piece of work from its initial description through implementation, review, and completion.

A typical workflow looks like this:

```text
Issue → Branch → Commits → Pull Request → Merge → Closed Issue
```

## Creating an Issue

A GitHub Issue is a record of a task or problem associated with a repository. Issues can be used to track:

- Bugs
- New features
- Improvements
- Documentation tasks
- Refactoring
- Research or investigation
- General project tasks

Suppose, for example, that you `my_ols` code has a problem. Instead of immediately changing the code, create an issue describing the problem first.

Open the repository on GitHub and select:

**Issues → New issue**

Give the issue a descriptive title:

```text
Wrong path in file opening code.
```

The description could contain:

```markdown
## Description

Running the script with relative paths causes the program to crash.

## Steps to reproduce

1. Create some input files locally.
2. Run the application.

## Expected behaviour

The script opens the files.

## Actual behaviour

Error message on the terminal, script crashes.

## Tasks

- [ ] Investigate the script file opening sections
- [ ] Fix the problem
- [ ] Add a regression test
- [ ] Verify
```

A good issue should explain **what needs to be done and why** without requiring someone to inspect the source code just to understand the task. After entering the information, submit the issue. GitHub assigns the issue a unique number, for example:

```text
#42
```

This number can be used to refer to the issue throughout the development workflow.

In addition, GitHub provides several mechanisms for organising and managing issues. You can assign people who are responsible for the task (Assignees), add labels (e.g. `bug`, `feature`, `documentation`), set up milestones (e.g. `version 2.0`) and break up the issue to sub-issues and dependencies.

## Creating a Branch

Development should normally not happen directly on the repository's default branch, such as `main`.

Instead, create a separate branch for the issue.

For issue `#42`, for example, we could create:

```text
42-file-opening
```

Using a separate branch has several advantages:

- `main` can remain stable.
- Multiple developers can work on different issues simultaneously.
- Changes belonging to an issue stay together.
- Changes can be reviewed before being merged.
- GitHub can associate the development branch with the issue.

A useful convention is to include the issue number in the branch name:

```text
42-file-opening
```

Another common convention is to include the type of work:

```text
fix/42-file-opening
feature/57-user-profile
docs/103-installation-guide
```

The exact convention is less important than using it consistently.

## Creating a Branch for an Issue

GitHub can create a development branch directly from an issue. Open the issue and locate the **Development** section in the right-hand sidebar. Select **Create a branch**. GitHub suggests a branch name, which you can change.

For issue `#42`, use:

```text
42-file-opening
```

Choose the appropriate repository and create the branch. Creating the branch from the issue has an important advantage: **GitHub automatically associates the branch with the issue.** The branch then appears in the issue's **Development** section.

>[!IMPORTANT]
> This association is useful because anyone reading the issue can immediately find the branch where the implementation is taking place.

> [!NOTE]
> GitHub can associate more than one branch or pull request with an issue when the work cannot reasonably be completed as a single change.

## Getting the Branch onto Your Computer

Since the repository has already been cloned, retrieve the latest branch information from GitHub:

```bash
git fetch origin
```

Then switch to the issue branch:

```bash
git switch 42-file-opening
```

If Git does not automatically create the corresponding local tracking branch, use:

```bash
git switch --track origin/42-file-opening
```

Check the current branch:

```bash
git branch
```

The output should look similar to:

```text
* 42-file-opening
  main
```

The `*` indicates the currently checked-out branch.

You can now implement the issue without modifying `main` directly.

> [!NOTE]
> A branch does not have to be created through the GitHub interface, you can create it locally using Git, and then you can associate it with the issue after pushing it to GitHub. 

## Working on the Issue

Make the required changes while working on the issue branch.

Check your current state with:

```bash
git status
```

After modifying the appropriate files, stage the changes:

```bash
git add .
```

Commit them:

```bash
git commit -m "Fix file opening issue"
```

Then push the commit to GitHub:

```bash
git push
```

Additional changes can be committed in the same way:

```bash
git add .
git commit -m "Add file opening regression test"
git push
```

The issue branch may therefore contain several commits implementing the same issue.

## Creating a Pull Request

Once the implementation is ready, create a pull request.

The pull request proposes merging the issue branch:

```text
42-file-opening
```

into:

```text
main
```

The development process now looks like this:

```text
Issue #42
    |
    v
42-file-opening
    |
    v
Pull Request
    |
    v
main
```

If the branch was created from the issue through GitHub, GitHub can associate the pull request with the issue as part of the development work.

## Linking a Pull Request to an Issue

The pull request should explicitly indicate which issue it resolves.

In the pull request description, add:

```text
Fixes #42
```

For example:

```markdown
## Summary

Fixes opening issues.

## Changes

- Corrected file opening
- Added error handling
- Added regression tests

Fixes #42
```

GitHub recognises closing keywords such as:

```text
Fixes #42
Closes #42
Resolves #42
```

When the pull request is merged into the repository's default branch, GitHub can automatically close the linked issue.

> [!IMPORTANT]
> Use a closing keyword only when the pull request actually completes the issue. If the pull request is merely related to an issue, reference `#42` without using `Fixes`, `Closes`, or `Resolves`.

The pull request becomes the place where the proposed implementation is reviewed.

## Merging and Closing the Issue

Once the pull request has been reviewed and approved, it can be merged into `main`.

If the pull request description contains:

```text
Fixes #42
```

the linked issue can automatically close when the pull request is merged into the default branch.

## Good Practices

> [!TIP]
> Keep Issues Focused. An issue should describe one reasonably coherent piece of work. If an issue becomes very large, consider dividing it into sub-issues.

>[!TIP]
>Use Descriptive Titles. Prefer: `File opening crashes with relative paths.` instead of: `File bug.` The title should make the problem understandable when someone views a list of issues.

>[!TIP]
> Define Completion Criteria. Use task lists or acceptance criteria to describe what must be completed before the issue can be considered resolved.

>[!TIP]
> Include Issue Numbers in Branch Names. Prefer: `42-file-opening` over an ambiguous branch name such as: `new-branch.` This makes the relationship between the branch and the tracked work immediately visible.

>[!TIP]
> Keep Unrelated Changes Separate. If you discover an unrelated bug while implementing issue `#42`, consider creating another issue and branch instead of adding the unrelated change to the existing branch. This keeps pull requests focused and easier to review.

>[!TIP]
> Link Development Work to Issues. Use GitHub's **Development** section to associate branches and pull requests with their corresponding issues.. This allows collaborators to move easily between: `Issue ↔ Branch ↔ Pull Request.`

>[!TIP]
> Use Pull Requests for Review. When working in a team, pull requests provide a useful review point before changes reach the default branch. They also preserve discussions about the implementation for future reference.

>[!TIP]
> Use Closing Keywords Carefully. Use: `Fixes #42` when the pull request completely resolves issue `#42`.

## Thanks

> [!IMPORTANT]
> Special thanks to Paul McCarthy (OxCIN, University of Oxford) for the codes.