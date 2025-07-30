---
name: Release checklist
about: 'Maintainers only: Checklist for making a new release'
title: 'Release vX.Y.Z'
labels: 'maintenance'
assignees: ''

---

**Zenodo DOI:**
https://doi.org/<ADD NEW DOI HERE>

<!-- Optional -->
**Target date:** YYYY/MM/DD

## Update dependencies

- [ ] Check all the dependency changes since the last release are  reflected in `environment.yml`
- [ ] Update the backup environment file `env/environment.yml` with `make conda_export`
- [ ] update PolarToolkit version in `environment.yml` in [PolarToolkit-Binder repo](https://github.com/mdtanker/polartoolkit-binder) to the latest version number

## Draft a Zenodo archive (to be done by a manager on Zenodo)

- [ ] Go to the [Zenodo entry](https://doi.org/10.5281/zenodo.7059091) for this project
- [ ] Create a "New version" of it.
- [ ] Get a new DOI for the new release
- [ ] Copy and paste the reserved DOI to this issue
- [ ] Update release date
- [ ] Update version number in Title (make sure there is a leading `v`, like `v1.5.7`)
- [ ] Update version number (use a leading `v` as well)
- [ ] Add as authors any new contributors who have added themselves to `AUTHORS.md` in the same order
- [ ] Save the release draft

## Update the changelog

- [ ] Generate a list of commits between the last release tag and now: `git log HEAD...v1.2.3 > changes.md`
- [ ] Use this to summarize the major changes to the code and add short descriptions to `CHANGELOG.md` (not `docs/changelog.md`!).
- [ ] Add the release date and Zenodo DOI badge to the top
- [ ] Add contributors to the list
- [ ] Open a PR to update the changelog
- [ ] Merge the PR

## Make a release

After the changelog PR is merged:

- [ ] Draft a new release on GitHub
- [ ] The tag and release name should be a version number (following Semantic Versioning) with a leading `v` (`v1.5.7`)
- [ ] Fill the release description with a Markdown version of the latest changelog entry (including the DOI badge)
- [ ] Publish the release

## Publish to Zenodo

- [ ] Upload the zip archive from the GitHub release to Zenodo
- [ ] Double check all information (date, authors, version)
- [ ] Publish the new version on Zenodo

## Conda-forge package

A PR should be opened automatically on the project feedstock repository.

- [ ] Add/remove/update any dependencies that have changed in `meta.yaml`
- [ ] If dropping/adding support for Python/numpy versions, make sure the correct version restrictions are applied in `meta.yaml`
- [ ] Merge the PR
