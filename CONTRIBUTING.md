# Contributing to NV Embedding Cache

If you are interested in contributing to NV Embedding Cache, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/nv-embedding-cache/issues)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
    - The team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the guidelines below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the guidelines below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Pull Requests
- Create pull requests targeting the main branch.
- Individual pull requests should be limited in scope and concise. Commit messages should be clear to facilitate code review.
Please do participate in code reviews to help improve the quality of the codebase. 
- Correctness tests are located in the tests/ folder. Please add tests, and/or modify existing tests, to cover the feature you are adding.
All tests should pass or the pull request will not be accepted. 
- Tests should also pass with [NVIDA Compute Sanitizer](https://developer.nvidia.com/compute-sanitizer), which is included with the CUDA Toolkit.
For more information about NVIDIA Compute Sanitizer, please refer to the [documentation](https://docs.nvidia.com/cuda/compute-sanitizer/index.html).
- Please add or update documentation for any new code or features both inline with the code as well as in the `README.md` files where applicable.
- If your change is likely to impact performance, please bring this up during the code review. 
- Please sign your commit using `git commit -s` or `--signoff` to certify that your work can be contributed to open source. 
- Please don't hesitate to open an issue and we will do our best to reply promptly. 

## Developer Certificate of Origin     

By signing your commit using `-s` or `--signoff`, you are certifying the following:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
