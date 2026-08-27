# Changelog

All user-relevant changes to this project are documented in this file.

## [1.0.2] - 2026-08-27

### Changed/hotfix

* Restructured the repository to remove `pandas` as a dependency for users of the public `sulib` package.

## [1.0.1] - 2026-08-27

### Added

* Added `scripts/example_public.py`, an example script for testing the public release.
* Added `sulib.su_decomp.generate_synthetic_pose_trajectory()` for generating synthetic pose trajectory data for quick numerical examples.
* Added this changelog.

### Changed

* Updated `sulib.su_decomp.RU()` to use NumPy's numerically more stable QR decomposition instead of Gram–Schmidt orthogonalization.

## [1.0.0] - 2026-08-26

### Added

* Initial public release of the `sulib` Python package.


